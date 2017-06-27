import logging
import random
import itertools
import numpy as np
import tensorflow as tf

from insta_pb2 import User

class Batcher(object):
  def __init__(self, hps, recordpath):
    self.recordpath = recordpath
    self.batch_size = hps.batch_size
    self.nfeats = hps.nfeats
    self.max_seq_len = hps.max_seq_len
    self.reset_record_iterator()

  def reset_record_iterator(self):
    self.records = tf.python_io.tf_record_iterator(self.recordpath)

  def get_batch(self, i):
    """(i currently ignored)"""
    bs = self.batch_size
    maxlen = self.max_seq_len
    x = np.zeros([bs, maxlen, self.nfeats])
    labels = np.zeros([bs, maxlen])
    seqlens = np.zeros([bs])
    lossmask = np.zeros([bs, maxlen])

    # TODO: hacky implementation. Right now just sample 1 sequence per user.
    for i in range(bs):
      user = User()
      try:
        user.ParseFromString(self.records.next())
      except StopIteration:
        logging.info("Starting a new 'epoch'. Resetting record iterator.")
        self.reset_record_iterator()
        user.ParseFromString(self.records.next())
      wrapper = UserWrapper(user)
      # TODO: incorporate lossmask
      x_i, l_i, s_i, lossmask_i = wrapper.sample_training_sequence(maxlen)
      x[i] = x_i
      labels[i] = l_i
      seqlens[i] = s_i
      lossmask[i] = lossmask_i
    return x, labels, seqlens, lossmask

def iterate_wrapped_users(recordpath):
  records = tf.python_io.tf_record_iterator(recordpath)
  for record in records:
    user = User()
    user.ParseFromString(record)
    yield UserWrapper(user)

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

  @property
  def seqlen(self):
    return len(self.user.orders) - 1

  @property
  def all_pids(self):
    pids = set()
    for order in self.user.orders[:-1]:
      pids.update( set(order.products) )
    return pids

  # TODO: should filter new products from last orders in the pb generation
  # step. then this won't be necessary.
  def last_order_predictable_prods(self):
    last_o = self.user.orders[-1]
    pids = self.all_pids
    last_pids = set(last_o.products)
    return pids.intersection(last_pids)

  # TODO: maybe these training_sequence methods should just return a 
  # big dataframe? This would solve the problem of having to return
  # these big tuples, and give us something for some downstream 
  # thing to work with for scaling or otherwise transforming features.
  # Also, for testing, it would make it easy to refer to features by name.
  def all_training_sequences(self, maxlen):
    pids = self.all_pids
    nseqs = len(pids)
    x = np.zeros([nseqs, maxlen, self.NFEATS])
    labels = np.zeros([nseqs, maxlen])
    lossmask = np.zeros([nseqs, maxlen])
    seqlens = np.zeros(nseqs) + self.seqlen
    for i, pid in enumerate(pids):
      # TODO: could be optimized to avoid redundant work
      x_i, l_i, _, lm_i = self.training_sequence_for_pid(pid, maxlen)
      x[i] = x_i
      labels[i] = l_i
      lossmask[i] = lm_i
    return x, labels, seqlens, lossmask


  def sample_pid(self):
    uniform = 1
    if uniform:
      pids = list(self.all_pids)
      return random.choice(pids)
    # Sample an order, then a product. This gives more probability weight to
    # frequently ordered products. Not clear whether that's a good thing.
    # Why -2? We don't want to include the last order, just because there might
    # be pids there for products never ordered in any previous order. A train
    # sequence focused on that product will give no signal.
    iorder = random.randint(0, self.norders-2)
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
    prevprev = None # The order before last
    # ids of products seen up to prev order (but not including products in prev
    # order itself)
    pids_seen = set()
    for i, prev, order in itertools.izip(range(maxlen), prevs, orders):
      ordered = pid in order.products
      labels[i] = int(ordered)
      lossmask[i] = int(seen_first)
      if ordered and not seen_first:
        seen_first = True
      previously_ordered = int(pid in prev.products)
      prevprods = set(prev.products)
      if prevprev is None:
        prev_repeats = 0
      else:
        prod2 = set(prevprev.products)
        prev_repeats = len(prevprods.intersection(prod2))
      # XXX: New feature
      prev_reorders = len(prevprods.intersection(pids_seen))
      x[i] = (
          [order.dow, order.hour, order.days_since_prior]
        + [previously_ordered, len(prev.products), prev_repeats, prev_reorders]
      )
      prevprev = prev
      pids_seen.update(set(prev.products))
    return x, labels, self.seqlen, lossmask

class Vectorizer(object):

  def vectorize(self, df, user):
    labels = df['label'].values
    lossmask = df['lossmask'].values
