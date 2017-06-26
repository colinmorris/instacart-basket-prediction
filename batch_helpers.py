import itertools
import numpy as np
import tensorflow as tf

from insta_pb2 import User

class Batcher(object):
  DEFAULT_RECORD_PATH = 'users.tfrecords'
  def __init__(self, hps, recordpath=None):
    if recordpath is None:
      recordpath = self.DEFAULT_RECORD_PATH
    self.batch_size = hps.batch_size
    self.nfeats = hps.nfeats
    self.max_seq_len = hps.max_seq_len
    self.reset_record_iterator()

  def reset_record_iterator(self):
    self.records = tf.python_io.tf_record_iterator(recordpath)

  def get_batch(self):
    bs = self.batchsize
    maxlen = self.max_seq_len
    x = np.zeros([bs, maxlen, self.nfeats])
    labels = np.zeros([bs, maxlen])
    seqlens = np.zeros([bs])

    for i in range(bs):
      user = User()
      try:
        user.ParseFromString(self.records.next())
      except StopIteration:
        self.reset_record_iterator()
        user.ParseFromString(self.records.next())
      wrapper = UserWrapper(user)
      x_i, l_i, s_i, lossmask_i = wrapper.sample_training_sequence(maxlen)
      x[i] = x_i
      labels[i] = l_i
      seqlens[i] = s_i
    return x, labels, seqlens


class UserWrapper(object):
  """Wrapper around User protobuf objs.
  """
  # 3 for this order, 3 for prev order
  # (day, hour, days_since) + (prev_ordered, prev_nprods, prev_nreorders)
  NFEATS = 3 + 3 

  def __init__(self, user):
    self.user = user

  @property
  def norders(self):
    return len(self.user.orders)

  def sample_pid(self):
    iorder = random.randint(0, self.norders-1)
    order = self.user.orders[iorder]
    iprod = random.randint(0, len(order.products)-1)
    return order.products[iprod]

  def sample_training_sequence(self, maxlen):
    pid = self.sample_pid()
    return self.training_sequence_for_pid(pid, maxlen)

  def training_sequence_for_pid(self, pid, maxlen):
    """Return a tuple of (x, labels, seqlen, lossmask)
    """
    nfeats = self.NFEATS
    x = np.zeros([maxlen, nfeats])
    labels = np.zeros([maxlen])
    lossmask = np.zeros([maxlen])
    # Start from second order (because of reasons)
    prevs, orders = itertools.tee(self.user.orders)
    orders.next()
    seen_first = False
    for i, prev, order in itertools.izip(range(maxlen), prevs, orders):
      ordered = pid in order.products
      labels[i] = int(ordered)
      lossmask[i] = int(seen_first)
      if ordered and not seen_first:
        seen_first = True
      previously_ordered = int(pid in prev.products)
      x[i] = (
          [order.dow, order.hour, order.days_since_prior]
          # TODO: implement prev_reorders feature
          # (Becomes easier once we treat pids as sets, which we probably should regardless)
        + [previously_ordered, len(prev.products), 0]
      )

