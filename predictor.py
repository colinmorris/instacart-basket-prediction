import pandas as pd
import tensorflow as tf

from scipy.special import expit

from constants import NONE_PRODUCTID

class BasePredictor(object):

  def __init__(self):
      pass

  def predict_next_order(self, history):
    """Return list of product_id for next order given a user's order
    history up to that point."""
    raise NotImplemented

class RnnModelPredictor(BasePredictor):

    def __init__(self, sess, model, thresh, predict_nones=True):
        self.model = model
        self.thresh = thresh
        self.sess = sess
        self.predict_nones = predict_nones
        assert model.hps.batch_size == 1

    def predict_prob(self, vec, seqlen):
        model = self.model
        sess = self.sess
        feed = {
            model.input_data:  [vec],
            model.sequence_lengths: [seqlen]
        }
        logits = sess.run(model.logits, feed)
        assert logits.shape == (1, 100), 'Got shape {}'.format(logits.shape)
        logits = logits[0] # Unwrap outer arr
        # TODO: when looking at these, why non-zero repeated values (e.g. -0.35435...)
        # past the 'end' of seq len? I guess that's the bias.
        logit = logits[seqlen-1]
        prob = expit(logit)
        return prob

    def predict_last_order(self, user):
      order = []
      tf.logging.debug('Calculating probabilities for user\'s {} reordered products'\
              .format(len(user.all_pids)))
      # Probability of no items being reordered
      p_none = 1
      for pid in user.all_pids:
        x, labels, seqlen, lossmask = user.training_sequence_for_pid(pid, 
            self.model.hps.max_seq_len)
        prob = self.predict_prob(x, seqlen)
        p_none *= (1 - prob)
        if prob >= self.thresh:
          order.append(pid)

      if self.predict_nones and p_none >= (1.5 * self.thresh): # XXX
        order.append(NONE_PRODUCTID)
      return order

class PreviousOrderPredictor(BasePredictor):

  def predict_last_order(self, user):
    return user.user.orders[-2].products

