from __future__ import division

import math
import numpy as np
from collections import namedtuple

Feature = namedtuple('feature', 'name arity fn')

FEATURES = []

def feature(fn):
  f = Feature(fn.__name__, 1, fn)
  FEATURES.append(f)
  return fn

def feature_with_arity(n):
  def inner_decorator(fn):
    f = Feature(fn.__name__, n, fn)
    FEATURES.append(f)
    return fn
  return inner_decorator

def define_passthrough_feature(colname):
  fn = lambda df, user: df[colname].values
  f = Feature(colname, 1, fn)
  FEATURES.append(f)

#define_passthrough_feature('days_since_prior')
define_passthrough_feature('previously_ordered')
define_passthrough_feature('n_prev_products')
define_passthrough_feature('n_prev_reorders')
define_passthrough_feature('n_prev_repeats')

@feature_with_arity(2)
def day_of_week_circular_sincos(df, user):
  # Raw value is an int from 0 to 6
  dow = df['dow'].values
  scaled = (dow / 7) * (2 * math.pi)
  sin = np.sin(scaled)
  cos = np.cos(scaled)
  return np.stack([sin, cos], axis=1)

@feature_with_arity(7)
def day_of_week_onehot(df, user):
  res = np.zeros([len(df), 7])
  res[np.arange(len(df)), df['dow'].values.astype(np.int8)] = 1
  return res

@feature_with_arity(2)
def hour_of_day_circular_sincos(df, user):
  # Raw value is an int from 0 to 6
  hr = df['hour'].values
  scaled = (hr / 24) * (2 * math.pi)
  sin = np.sin(scaled)
  cos = np.cos(scaled)
  return np.stack([sin, cos], axis=1)

@feature
def prev_repeat_rate(df, user):
  # What proportion of last order was repeated?
  sizes = df['n_prev_products'].values # Should always be non-zero... right?
  repeats = df['n_prev_repeats'].values
  return repeats / sizes

@feature
def prev_reorder_rate(df, user):
  # What proportion of last order was reordered
  sizes = df['n_prev_products'].values # Should always be non-zero... right?
  repeats = df['n_prev_reorders'].values
  return repeats / sizes

@feature
def days_since_is_maxed(df, user):
  # Is days_since_prior_order its max value (30)? (It's pretty clear that
  # this variable is truncated to that max)
  return (df['days_since_prior'].values == 30).astype(int)
@feature
def same_day(df, user):
  return (df['days_since_prior'].values == 0).astype(int)

@feature
def days_since_last_scaled(df, user):
  days = df['days_since_prior'].values
  return days / 30

NFEATS = sum(f.arity for f in FEATURES)
