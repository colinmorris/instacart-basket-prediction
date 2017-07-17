from __future__ import division

import pytest
import math
import os
import numpy as np
import numpy.testing as npt

from baskets.insta_pb2 import User
from baskets import batch_helpers
from baskets.batch_helpers import UserWrapper
from baskets import features as feats
from baskets import common
from baskets.test_helpers import user

TEST_UID = 2455
userinfo = dict(
    norders= 7,
    nprodorders=31,
    # Lots of milk.
    # Recorded in approximately chrono order
    prods=[15598, 27845, 35951, 15429, 27086, 19348,
      33334,
      14233,
      47114,
      27243, 22950, 30066, 32423, 49424,
      # NB: These last two are unique to the final order
      47630, 26209],
)
HHID = 27086 # pid for half-and-half

def test_userwrapper_props(user):
  assert user.norders == userinfo['norders']
  prods = set(userinfo['prods'])
  trainable_prods = prods.difference([47630, 26209])
  theirprods = user.all_pids
  assert len(theirprods) == len(trainable_prods)
  assert theirprods == trainable_prods

def test_half_and_half_trainseq(user):
  maxlen = 100
  ts = user.training_sequence_for_pid(HHID, maxlen)
  assert ts['pindex'] == HHID-1 # Translated from 1-indexing to 0-indexing
  # Training sequence starts from second order
  seqlen = ts['seqlen']
  assert seqlen == user.norders - 1
  hh_orders = np.array([0, 3, 4, 6])
  true_label_indices = np.array([3, 4, 6]) - 1
  expected_labels = np.zeros(maxlen)
  expected_labels[true_label_indices] = 1
  assert (ts['labels'] == expected_labels).all()

  lossmask = ts['lossmask']
  assert (lossmask[seqlen:] == 0).all()
  assert (lossmask[:seqlen] == 1).all()

def test_half_and_half_features(user):
  maxlen = 100
  ts = user.training_sequence_for_pid(HHID, maxlen)
  seqlen = ts['seqlen']
  x = ts['x']
  xraw = ts['xraw']
  df = user.rawfeats_to_df(xraw)

  assert len(df) == seqlen
  days_since = [2, 1, 3, 2, 1, 1]
  assert (df['days_since_prior'] == days_since).all()
  hours = [7, 10, 7, 8, 8, 15]
  assert (df['hour'].values == hours).all()
  
  hh_orders = np.array([0, 3, 4, 6])
  hh_indices = [0, 3, 4, 6]
  # Above are wrt all orders. Because train seqs start from 2nd order, we need to subtract
  # 1 to get corresponding sequence indices. Then we *add* 1 to get the indices
  # of the orders for which these are the previous orders.
  prev_ordered_indices = [0, 3, 4] # we drop 6 because it falls off the end
  # NB: the order of products shown in snooping.ipynb is NOT the true
  # add to cart order.
  prev_ordered_addtocart_order = [2, 1, 5]
  prev_ordered = np.zeros(seqlen)
  prev_ordered[prev_ordered_indices] = prev_ordered_addtocart_order
  assert (df['previously_ordered'].values == prev_ordered).all()
  
  products_per_order = [6, 5, 2, 3, 6, 5] # (last order = 4, but not used)
  assert (df['n_prev_products'] == products_per_order).all()

  reorders = [0, 4, 1, 3, 5, 0, 2][:-1]
  repeats =  [0, 4, 1, 0, 3, 0, 0][:-1]
  assert (df['n_prev_reorders'].values == reorders).all()
  assert (df['n_prev_repeats'].values == repeats).all()
  # TODO: other raw feats

  vec = batch_helpers.vectorize(df, user.user, maxlen)
  assert vec.shape == (maxlen, feats.NFEATS)
  prev_ordered_padded = np.pad(prev_ordered, (0, maxlen-seqlen), 'constant')
  # New 0th feature is binary yes/no was in prev order
  prev_ordered_padded = prev_ordered_padded > 0
  assert (vec[:,0] == prev_ordered_padded).all()

  vec_n_prev_products = vec[:,1]
  assert (vec_n_prev_products[:seqlen] > 0).all()
  assert (vec_n_prev_products[:seqlen] == products_per_order).all()
  assert (vec_n_prev_products[seqlen:] == 0).all()

  dow_sincos = feats.day_of_week_circular_sincos(df, user)
  assert dow_sincos.shape == (seqlen, 2)
  dows = [2, 4, 5, 1, 3, 4, 5]
  # skip first order
  dows = dows[1:]
  sin, cos = dow_sincos[0]
  scaled = (4/7.0) * (2 * math.pi)
  assert sin == np.sin(scaled)
  assert cos == np.cos(scaled)

  assert (dow_sincos[0] != dow_sincos[1]).all()
  assert (dow_sincos >= -1).all()
  assert (dow_sincos <= 1).all()

  dow_onehot = feats.day_of_week_onehot(df, user)
  assert dow_onehot.shape == (seqlen, 7)
  npt.assert_array_equal(dow_onehot[0], [0, 0, 0, 0, 1, 0, 0])
  #assert (dow_onehot[0] == [0, 0, 1, 0, 0, 0, 0]).all()
  assert (dow_onehot.sum(axis=1) == [1, 1, 1, 1, 1, 1]).all()

  rep_rate = feats.prev_repeat_rate(df, user)
  reorder_rate = feats.prev_reorder_rate(df, user)
  # For the first sequence element, the prev order was the user's first one,
  # so it will have had no repeats/reorders.
  assert rep_rate[0] == 0
  assert reorder_rate[0] == 0
  # Order #2 has 4 products that were in order #1 and 1 new one.
  assert rep_rate[1] == 4/5
  assert reorder_rate[1] == 4/5
  assert rep_rate[2] == reorder_rate[2] == 1/2
  # Order #4 has reorders but no repeats
  assert rep_rate[3] == 0
  assert reorder_rate[3] == 1

  maxdays = feats.days_since_is_maxed(df, user)
  sameday = feats.same_day(df, user)
  assert not maxdays.any()
  assert not sameday.any()

  days_scaled = feats.days_since_last_scaled(df, user)
  days_since = np.array([2, 1, 3, 2, 1, 1])
  assert (days_scaled == days_since/30).all()
  assert days_scaled[0] > days_scaled[1]
  
  
  prev_ordered_indices = [0, 3, 4] # we drop 6 because it falls off the end
  prev_ordered_addtocart_order = [2, 1, 5]

  prevfirst = feats.previously_firstordered(df, user)
  assert prevfirst.sum() == 1
  assert prevfirst[3] == 1
  prev_firsthalf = feats.previously_ordered_firsthalf(df, user)
  assert (prev_firsthalf == [1, 0, 0, 1, 0, 0]).all()
  prev_secondhalf = feats.previously_ordered_secondhalf(df, user)
  assert (prev_secondhalf == [0, 0, 0, 0, 1, 0]).all()










