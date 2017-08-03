from __future__ import division
import logging
import numpy as np

from scipy.special import expit

from baskets.constants import NONE_PRODUCTID
import baskets.fscore as fscore_helpers
import baskets.clever_fscore as clever
from baskets.cleverest_fscore import F1Optimizer

class BasePredictor(object):

  def __init__(self):
    pass

  def predict_last_order(self, user):
    """Return list of product_id for last order"""
    raise NotImplemented

class PreviousOrderPredictor(BasePredictor):
  def predict_last_order(self, user):
    return user.user.orders[-2].products

class MissingProbsException(Exception):
  pass

class ProbabilisticPredictor(BasePredictor):

  def __init__(self, probmap):
    super(ProbabilisticPredictor, self).__init__()
    self.probmap = probmap

  def predict_last_order(self, user):
    try:
      pid_to_prob = self.probmap[user.uid]
    except KeyError:
      msg = 'Missing probabilities for user {} in pdict of length {}'.format(
          user.uid, len(self.probmap))
      raise MissingProbsException(msg)
    reorders = self.predict_order_from_probs(pid_to_prob)
    return reorders

  def predict_order_from_probs(self, pid_to_prob):
    raise NotImplemented

  @classmethod
  def predict_order_by_threshold(kls, pid_to_prob, thresh):
    reorders = [pid for (pid, prob) in pid_to_prob.iteritems() if prob >= thresh]
    probs = np.array(pid_to_prob.values())
    p_none = np.product(1-probs)
    if p_none > thresh:
      reorders.append(NONE_PRODUCTID)
    return reorders

class ThresholdPredictor(ProbabilisticPredictor):

  def __init__(self, probmap, thresh):
    super(ThresholdPredictor, self).__init__(probmap)
    self.thresh = thresh

  def predict_order_from_probs(self, pid_to_prob):
    return self.predict_order_by_threshold(pid_to_prob, self.thresh)

class MonteCarloThresholdPredictor(ProbabilisticPredictor):

  def __init__(self, probmap, ntrials=80, save=False, optimization_level=10):
    super(MonteCarloThresholdPredictor, self).__init__(probmap)
    self.ntrials = ntrials
    self.save = save
    self.optimization_level = optimization_level
    self.history = []

  def predict_order_from_probs(self, pid_to_prob):
    if self.optimization_level == 0 and len(pid_to_prob) > 200:
      logging.warning('Falling back to static threshold for user with {} products'.format(
        len(pid_to_prob)))
      thresh = .2
      return self.predict_order_by_threshold(pid_to_prob, thresh)
    probs = np.array(pid_to_prob.values())
    probs = np.sort(probs)[::-1]
    best_k, predict_none, _maxf1 = F1Optimizer.maximize_expectation(probs)
    try:
      thresh = probs[best_k]
    except IndexError:
      # 'Off the charts'. Probs are sorted in descending order. best_k = len(probs)
      # means 'predict all the things'
      thresh = -1
    prods = [pid for pid,prob in pid_to_prob.iteritems() if prob > thresh]
    if predict_none:
      prods.append(NONE_PRODUCTID)
    return prods

  # XXX: deprecated. Based on an f-score optimization algorithm that got superseded
  # by a faster one someone posted as a kernel on kaggle.
  def _predict_order_from_probs(self, pid_to_prob):
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
    pred = self.predict_order_by_threshold(pid_to_prob, best_seen[0])
    if self.save:
      pred_evt = dict(thresh=best_seen[0], predicted=pred)
      self.history.append(pred_evt)
    return pred

  def evaluate_threshold(self, thresh, probs):
    return fscore_helpers.expected_fscore_montecarlo(probs, thresh, self.ntrials)

  def get_candidate_thresholds(self, probs, exhaustive=False):
    if exhaustive:
      yield -1, False
    minprob = .05
    maxprob = .5
    # Not worth the time savings to skip any threshes in between.
    # A small threshold difference can have a surprisingly large
    # effect on E[f]. One example I've personally witnessed:
    # E[f; thresh=.106] = .42
    # E[f; thresh=.107] = .47
    mindelta = .005 if self.optimization_level == 0 else 0
    lastprob = -1
    for prob in probs:
      if not (minprob <= prob <= maxprob):
        if exhaustive:
          yield prob, False
        continue
      delta = abs(lastprob - prob)
      if delta <= mindelta:
        if exhaustive:
          yield prob, False
        continue
      yield (prob, True) if exhaustive else prob
      lastprob = prob
    # End on a big one. All the thresholds we've yielded so far have been
    # low enough to allow at least one product in. It could be that the 
    # optimal thresh is higher than the max product prob (i.e. the optimal
    # strategy is just to predict none)
    if exhaustive:
      yield (1.01, True)
    else:
      yield .5

class HybridThresholdPredictor(MonteCarloThresholdPredictor):
  """Calculate exact expected fscore for users with small number
  of total products. Use mc simulation for others.
  """
  # 100 products: .1s
  # 200: 1s
  # 300: 3.5s
  # 400: 9s
  # 500: 20s
  # XXX: May want to increase this when generating final submission file, 
  # if you're really patient.
  MAX_PRODUCTS_FOR_EXACT = 300

  @property
  def too_many_products(self):
    return self.MAX_PRODUCTS_FOR_EXACT / 3 if self.optimization_level == 0 else\
        self.MAX_PRODUCTS_FOR_EXACT

  def evaluate_threshold(self, thresh, probs):
    if len(probs) > self.too_many_products:
      return fscore_helpers.expected_fscore_montecarlo(probs, thresh, self.ntrials)
    else:
      return clever.efscore(probs, thresh)

