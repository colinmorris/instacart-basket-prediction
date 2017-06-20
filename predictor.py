import pandas as pd
import tensorflow as tf

from scipy.special import expit

from dataset import Dataset

class BasePredictor(object):

  def __init__(self):
      pass

  def predict_next_order(self, history):
    """Return list of product_id for next order given a user's order
    history up to that point."""
    raise NotImplemented

class RnnModelPredictor(BasePredictor):

    def __init__(self, sess, model, thresh):
        self.model = model
        self.thresh = thresh
        self.sess = sess
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

    def predict_next_order(self, history):
        labels, vecs, seqlens, pids = Dataset.convert_df(history, self.model.hps.max_seq_len)
        tf.logging.debug('Calculating probabilities for user\'s {} reordered products'\
                .format(len(pids)))
        order = []
        for (i, pid) in enumerate(pids):
            prob = self.predict_prob(vecs[i], seqlens[i])
            if prob >= self.thresh:
                order.append(pid)
        return order

class PreviousOrderPredictor(BasePredictor):

  def __init__(self, db, always_none=0):
    super(PreviousOrderPredictor, self).__init__()
    self.db = db
    self.always_none = always_none

  def predict_order(self, order):
    """Return a series of product ids"""
    uid = order.user_id
    onum = order.order_number
    assert onum > 1, onum
    orders = self.db.orders
    prevorder = orders[
        (orders['user_id'] == uid)
        & (orders['order_number'] == onum-1)
        ]
    assert len(prevorder) == 1
    prevorder = prevorder.iloc[0]
    prevprods = self.db.get_ops(prevorder)
    prods = prevprods['product_id']
    if self.always_none:
      prods = prods.append(pd.Series([0]))
    return prods
