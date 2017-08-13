import random
import pandas as pd
import numpy as np
import itertools
import tensorflow as tf
import scipy.special

from baskets.insta_pb2 import User
from baskets import common, constants

class UserWrapper(object):
  """Wrapper around User protobuf objs.
  """
  # 3 for this order, 3 for prev order
  # (day, hour, days_since) + (prev_ordered, prev_nprods, prev_nreorders)
  NFEATS = 3 + 3 

  def __init__(self, user, ktest=False):
    # (Really this should be self._user. Clients discouraged from using this attr directly.)
    self.user = user
    self._all_pids = None
    self.ktest = ktest
    if ktest:
      assert self.user.test

  @property
  def uid(self):
    return self.user.uid

  @property
  def orders(self):
    orders = self.user.orders
    if self.ktest:
      # Maybe cache this property if it's being accessed a lot?
      orders = list(orders)
      orders.append(self.user.testorder)
    return orders

  @property
  def norders(self):
    return len(self.user.orders)

  @property
  def nprods(self):
    return len(self.all_pids)

  @property
  def seqlen(self):
    # Minus the first order, which is never a training example
    return len(self.orders) - 1

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
      for order in self.orders[:-1]:
        pids.update( set(order.products) )
      self._all_pids = pids
    return self._all_pids

  @property
  def sorted_pids(self):
    return sorted(self.all_pids)

  # (used by libfm)
  def order_pairs(self):
    """Return tuples of (prev_order, order)
    """
    a, b = itertools.tee(self.orders)
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

# deprecated
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

def iterate_wrapped_users(recordpath, ktest=False):
  recordpath = common.resolve_recordpath(recordpath)
  records = tf.python_io.tf_record_iterator(recordpath)
  for record in records:
    user = User()
    user.ParseFromString(record)
    yield UserWrapper(user, ktest)

def canonical_ordered_uid_pids(fold):
  users = iterate_wrapped_users(fold)
  for user in users:
    for pid in user.sorted_pids:
      yield user.uid, pid

# TODO: nice to be able to pass multiple tags
def logits_for_tag(tag, fold):
  pdict = common.pdict_for_tag(tag, fold)
  logits = []
  for (uid, pid) in canonical_ordered_uid_pids(fold):
    prob = pdict[uid][pid]
    logit = scipy.special.logit(prob)
    logits.append(logit)

  return logits
