import pandas as pd
import tensorflow as tf

from scipy.special import expit

from constants import NONE_PRODUCTID
import fscore as fscore_helpers

class BasePredictor(object):

  def __init__(self):
    pass

  def predict_last_order(self, user):
    """Return list of product_id for last order"""
    raise NotImplemented

class PreviousOrderPredictor(BasePredictor):
  def predict_last_order(self, user):
    return user.user.orders[-2].products

class ProbabilisticPredictor(BasePredictor):

  def __init__(self, probmap):
    super(ProbabilisticPredictor, self).__init__()
    self.probmap = probmap

  def predict_last_order(self, user):
    pid_to_prob = self.probmap[user.uid]
    reorders = self.predict_order_from_probs(pid_to_prob)
    return reorders

  def predict_order_from_probs(self, pid_to_prob):
    raise NotImplemented

  @classmethod
  def predict_order_by_threshold(kls, pid_to_prob, thresh):
    return [pid for (pid, prob) in pid_to_prob.iteritems() if prob >= thresh]

class ThresholdPredictor(ProbabilisticPredictor):

  def __init__(self, probmap, thresh):
    super(ThresholdPredictor, self).__init__(probmap)
    self.thresh = thresh

  def predict_order_from_probs(self, pid_to_prob):
    return self.predict_order_by_threshold(pid_to_prob, self.thresh)

class MonteCarloThresholdPredictor(ProbabilisticPredictor):

  def __init__(self, probmap, ntrials):
    super(MonteCarloThresholdPredictor, self).__init__(probmap)
    self.ntrials = ntrials

  def predict_order_from_probs(self, pid_to_prob):
    items = pid_to_prob.items()
    # Sort on probability
    items.sort(key = lambda i: i[1])
    pids = [i[0] for i in items]
    probs = [i[1] for i in items]
    # get canddiate thresholds
    thresh_cands = self.get_candidate_thresholds(probs)
    # TODO: rather than just returning the threshold that gives the highest fscore, we might
    # do better to incorporate some prior about the smoothness of fn from thresh to fscore.
    # e.g. if we see something like
    # {.15: .25, .16: .24, .17: .29, .18: .24, ... .21: .26, .22: .27, .23: .28, .24: .26, ...}
    # then maybe we should pick a thresh of .23 rather than .17, which might just have been a fluke
    # TODO: at some point should make an ipython notebook to explore this stuff and make 
    # some graphs
    best_seen = (None, -1)
    thresh_scores = [] # for debugging
    for thresh in thresh_cands:
      fscore = self.evaluate_threshold(thresh, probs)
      if fscore > best_seen[1]:
        best_seen = (thresh, fscore)
      thresh_scores.append( (thresh, fscore) )
      
    # return predictions according to best thresh
    return self.predict_order_by_threshold(pid_to_prob, best_seen[0])

  def evaluate_threshold(self, thresh, probs):
    return fscore_helpers.expected_fscore_montecarlo(probs, thresh, self.ntrials)

  def get_candidate_thresholds(self, probs):
    minprob = .05
    maxprob = .5
    mindelta = .005
    lastprob = -1
    for prob in probs:
      if not (minprob <= prob <= maxprob):
        continue
      delta = abs(lastprob - prob)
      if delta < mindelta:
        continue
      yield prob
      lastprob = prob
    # End on a big one. All the thresholds we've yielded so far have been
    # low enough to allow at least one product in. It could be that the 
    # optimal thresh is higher than the max product prob (i.e. the optimal
    # strategy is just to predict none)
    yield .5

class HybridThresholdPredictor(MonteCarloThresholdPredictor):
  """Calculate exact expected fscore for users with small number
  of total products. Use mc simulation for others.
  """
  # Of 509 users in test set, 50 have <= 13 prods
  MAX_PRODUCTS_FOR_EXACT = 13 # less than .5s 
  def evaluate_threshold(self, thresh, probs):
    if len(probs) > self.MAX_PRODUCTS_FOR_EXACT:
      return fscore_helpers.expected_fscore_montecarlo(probs, thresh, self.ntrials)
    else:
      return fscore_helpers.exact_expected_fscore_naive(probs, thresh)

## Stuff below is basically deprecated

class BaseRNNModelPredictor(BasePredictor):
  """Parent class for predictors that generate predictions from a 'live' RNN
  model. Not really used anymore for eval, but kept around for compat with poke_eval.py
  """
  def __init__(self, sess, model, **kwargs):
    self.model = model
    self.sess = sess
    assert model.hps.batch_size == 1

  def _get_logits(self, vec, seqlen, pindex):
    model = self.model
    sess = self.sess
    # TODO: use model_helpers feed dict helper
    feed = {
        model.input_data:  [vec],
        model.sequence_lengths: [seqlen]
    }
    if model.hps.product_embeddings:
      feed[model.product_ids] = [pindex] 
    logits = sess.run(model.logits, feed)
    assert logits.shape == (1, 100), 'Got shape {}'.format(logits.shape)
    logits = logits[0] # Unwrap outer arr
    # TODO: when looking at these, why non-zero repeated values (e.g. -0.35435...)
    # past the 'end' of seq len? I guess that's the bias.
    return logits

  def predict_prob(self, vec, seqlen, pindex):
    logits = self._get_logits(vec, seqlen, pindex)
    logit = logits[seqlen-1]
    prob = expit(logit)
    return prob

  @classmethod
  def predict_order_by_threshold(kls, pid_to_prob, thresh):
    order = []
    # Probability of no items being reordered
    p_none = 1
    for pid, prob in pid_to_prob.iteritems():
      p_none *= (1 - prob)
      if prob >= thresh:
        order.append(pid)

    # TODO: refactor this out
    if 0 and self.predict_nones and p_none >= (1.5 * thresh): # XXX
      order.append(NONE_PRODUCTID)
    return order

  def predict_last_order(self, user):
    tf.logging.debug('Calculating probabilities for user\'s {} reordered products'\
            .format(len(user.all_pids)))
    pid_to_prob = {}
    for pid in user.all_pids:
      ts = user.training_sequence_for_pid(pid, self.model.hps.max_seq_len)
      prob = self.predict_prob(ts['x'], ts['seqlen'], ts['pindex'])
      pid_to_prob[pid] = prob
    return self.predict_last_order_from_probs(pid_to_prob)

class RnnModelPredictor(BaseRNNModelPredictor):

  def __init__(self, sess, model, thresh, predict_nones=True):
    self.model = model
    self.thresh = thresh
    self.sess = sess
    self.predict_nones = predict_nones
    assert model.hps.batch_size == 1

  def predict_last_order_from_probs(self, pid_to_prob):
    return self.predict_order_by_threshold(pid_to_prob, self.thresh)

class MonteCarloRnnPredictor(BaseRNNModelPredictor):
  DEFAULT_NTRIALS = 20
  def __init__(self, *args, **kwargs):
    super(MonteCarloRnnPredictor, self).__init__(*args, **kwargs)
    self.ntrials = kwargs.get('ntrials', self.DEFAULT_NTRIALS)

  def predict_last_order_from_probs(self, pid_to_prob):
    items = pid_to_prob.items()
    # Sort on probability
    items.sort(key = lambda i: i[1])
    pids = [i[0] for i in items]
    probs = [i[1] for i in items]
    # get canddiate thresholds
    thresh_cands = self.get_candidate_thresholds(probs)
    # TODO: rather than just returning the threshold that gives the highest fscore, we might
    # do better to incorporate some prior about the smoothness of fn from thresh to fscore.
    # e.g. if we see something like
    # {.15: .25, .16: .24, .17: .29, .18: .24, ... .21: .26, .22: .27, .23: .28, .24: .26, ...}
    # then maybe we should pick a thresh of .23 rather than .17, which might just have been a fluke
    # TODO: at some point should make an ipython notebook to explore this stuff and make 
    # some graphs
    best_seen = (None, -1)
    thresh_scores = [] # for debugging
    for thresh in thresh_cands:
      fscore = fscore_helpers.expected_fscore_montecarlo(probs, thresh, self.ntrials)
      if fscore > best_seen[1]:
        best_seen = (thresh, fscore)
      thresh_scores.append( (thresh, fscore) )
      
    # return predictions according to best thresh
    return self.predict_order_by_threshold(pid_to_prob, best_seen[0])

  def get_candidate_thresholds(self, probs):
    # TODO: Possible ways to be more clever: 
    # - impose hard limits on min/max thresholds to test
    # - ignore small deltas
    for prob in probs:
      yield prob

