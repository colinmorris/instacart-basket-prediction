import random
import pandas as pd
import numpy as np
import itertools
import tensorflow as tf

from baskets.insta_pb2 import User
from baskets import common, constants

class UserWrapper(object):
  """Wrapper around User protobuf objs.
  """
  # 3 for this order, 3 for prev order
  # (day, hour, days_since) + (prev_ordered, prev_nprods, prev_nreorders)
  NFEATS = 3 + 3 

  def __init__(self, user, feat_fixture=None):
    self.user = user
    self._all_pids = None
    self.feat_fixture = feat_fixture

  @property
  def uid(self):
    return self.user.uid

  @property
  def norders(self):
    return len(self.user.orders)

  @property
  def nprods(self):
    return len(self.all_pids)

  @property
  def seqlen(self):
    return len(self.user.orders) - 1

  @property
  def istest(self):
    return self.user.test

  @property
  def all_pids(self):
    """Return a set of ids of all products occurring in orders up to but not
    including the final one."""
    # This can get called a fair number of times, so cache it
    if self._all_pids is None:
      pids = set()
      for order in self.user.orders[:-1]:
        pids.update( set(order.products) )
      self._all_pids = pids
    return self._all_pids

  def order_pairs(self):
    """Return tuples of (prev_order, order)
    """
    a, b = itertools.tee(self.user.orders)
    b.next()
    return itertools.izip(a, b)

  # TODO: should filter new products from last orders in the pb generation
  # step. then this won't be necessary.
  def last_order_predictable_prods(self):
    last_o = self.user.orders[-1]
    pids = self.all_pids
    last_pids = set(last_o.products)
    res = pids.intersection(last_pids)
    # Special case: if predictable prods is empty, return a single dummy
    # product id representing "none" (to be consistent with kaggle's scoring method)
    if not res:
      return set([constants.NONE_PRODUCTID])
    return res

  def sample_pids(self, n):
    if n > len(self.all_pids):
      return self.all_pids
    return random.sample(self.all_pids, n)

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

  # TODO: clear out dusty old methods

  rawcols = ['dow', 'hour', 'days_since_prior',
      'previously_ordered', 'n_prev_products', 'n_prev_repeats', 'n_prev_reorders']
  def rawfeats_to_df(self, featdata):
    return pd.DataFrame(featdata, columns=self.rawcols)
  def transform_raw_feats(self, featdata, maxlen): 
    df = self.rawfeats_to_df(featdata)
    if self.feat_fixture:
      feats, dimen = self.feat_fixture
      return vectorize(df, self.user, maxlen, feats, dimen)
    else:
      return vectorize(df, self.user, maxlen)

  def training_sequence_for_pid(self, pid, maxlen, product_df=None, testmode=False):
    """Return a dict of (x, labels, seqlen, lossmask, pid)
    Where x is an ndarray, seqlen and pid are scalars, and everything else is a 1-d array

    In test mode, include the final order (which will have unknown label)
    """
    seqlen = self.seqlen
    if testmode:
      assert self.user.test
      seqlen += 1
    x = np.zeros([seqlen, len(self.rawcols)])
    labels = np.zeros([maxlen])
    lossmask = np.zeros([maxlen])
    # Start from second order (because of reasons)
    # This is pretty hacky. Make a copy of the current user, and add their 
    # "testorder" to the end of their array of normal orders.
    user = self.user
    if testmode:
      user = User()
      user.CopyFrom(self.user)
      lastorder = user.orders.add()
      lastorder.CopyFrom(self.user.testorder)
      assert len(user.orders) == self.norders + 1
    prevs, orders = itertools.tee(user.orders)
    orders.next()
    seen_first = False
    prevprev = None # The order before last
    # ids of products seen up to prev order (but not including products in prev
    # order itself)
    pids_seen = set()
    for i, prev, order in itertools.izip(range(maxlen), prevs, orders):
      ordered = pid in order.products
      prevprods_in_order = list(prev.products)
      try:
        # Order in which it was added to the cart (first=1)
        previously_ordered = prevprods_in_order.index(pid) + 1
      except ValueError:
        previously_ordered = 0
      labels[i] = int(ordered)
      seen_first = seen_first or previously_ordered
      # We only care about our ability to predict when a product is *re*ordered. 
      # So we zero out the loss for predicting all labels up to and including the
      # first order that has that product. (We also zero out the loss past the end
      # of the actual sequence, i.e. for the padding section)
      lossmask[i] = int(bool(seen_first))
      prevprods = set(prev.products)
      if prevprev is None:
        prev_repeats = 0
      else:
        prod2 = set(prevprev.products)
        prev_repeats = len(prevprods.intersection(prod2))
      prev_reorders = len(prevprods.intersection(pids_seen))
      x[i] = (
          [order.dow, order.hour, order.days_since_prior]
        + [previously_ordered, len(prev.products), prev_repeats, prev_reorders]
      )
      prevprev = prev
      pids_seen.update(set(prev.products))

    feats = self.transform_raw_feats(x, maxlen)
    res = dict(x=feats, labels=labels, seqlen=seqlen, lossmask=lossmask,
        pindex=pid-1, xraw=x
    )
    if product_df is not None:
      aid, did = product_df.loc[pid-1, ['aisle_id', 'department_id']]
      res['aisle_id'] = aid-1
      res['dept_id'] = did-1
    return res

def vectorize(df, user, maxlen, features=None, nfeats=None):
  features = features or FEATURES # Default to all of them
  nfeats = nfeats or NFEATS
  res = np.zeros([maxlen, nfeats])
  i = 0
  seqlen = len(df) 
  for feat in features:
    featvals = feat.fn(df, user)
    if feat.arity == 1:
      res[:seqlen,i] = featvals
    else:
      res[:seqlen,i:i+feat.arity] = featvals
    i += feat.arity
  return res

def iterate_wrapped_users(recordpath):
  recordpath = common.resolve_recordpath(recordpath)
  records = tf.python_io.tf_record_iterator(recordpath)
  for record in records:
    user = User()
    user.ParseFromString(record)
    yield UserWrapper(user)
