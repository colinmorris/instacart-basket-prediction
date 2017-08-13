from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import math_ops
from collections import namedtuple

_raw_feats = ['previously_ordered', 'days_since_prior', 'dow', 'hour',
      'n_prev_products', ]
      #'n_prev_repeats', 'n_prev_reorders']

Feature = namedtuple('Feature', 'name arity fn binary')

ALL_FEATURES = []
FEAT_LOOKUP = {}

def _normalize_feat_output(output_tensor, arity, name):
  rank = len(output_tensor.shape)
  if rank == 1:
    output_tensor = tf.expand_dims(output_tensor, 0)
  shape = output_tensor.shape
  assert shape[0] == arity, "Expected shape of {} to be ({}, ?), but was {}".format(name, arity, shape)
  rank = len(shape)
  assert rank == 2, "Expected rank 2, got {}".format(rank)
  if output_tensor.dtype != tf.float32:
    output_tensor = tf.cast(output_tensor, tf.float32)
  return output_tensor

def feature(arity=1, name=None, keys=None, binary=False):
  assert isinstance(arity, int)
  def decorator(featfn):
    argnames = keys or featfn.func_code.co_varnames
    def wrapped(dataset):
      args = [dataset[arg] for arg in argnames]
      res = featfn(*args)
      return _normalize_feat_output(res, arity, name)

    featname = name or featfn.__name__
    feat = Feature(featname, arity, wrapped, binary)
    ALL_FEATURES.append(feat)
    assert featname not in FEAT_LOOKUP
    FEAT_LOOKUP[featname] = feat
    return wrapped

  return decorator

def define_passthrough_feature(key):
  @feature(name=key, keys=(key,), arity=1)
  def _inner(val):
    return val

# TODO: probably should apply a log transform to most of these counts or something
_passthrough_keys = ['days_since_prior', 'n_prev_products',
    'n_prev_reorders', 'n_prev_repeats',
    ]
for k in _passthrough_keys:
  define_passthrough_feature(k)

# boundaries=[0, 1, 2] leads to buckets (-inf, 0), [0, 1), [1, 2), [2, inf)
def define_bucketized_feature(key, boundaries):
  name = key + '_bucketized'
  boundaries = map(float, boundaries)
  nbuckets = len(boundaries) + 1
  @feature(arity=nbuckets, name=name, keys=[key], binary=True)
  def _innerfeat(val):
    bucket_indices = math_ops._bucketize(val, boundaries=boundaries)
    return tf.one_hot(bucket_indices, depth=nbuckets, axis=0)

def define_onehot_feature(key, depth):
  @feature(arity=depth, name=key+'_onehot', keys=[key], binary=True)
  def _innerfeat(val):
    return tf.one_hot(tf.cast(val, tf.int64), depth, axis=0)

define_onehot_feature('dow', 7)
#define_onehot_feature('hour', 24)

hour_buckets = [7, 10, 13, 16, 20,]
define_bucketized_feature('hour', hour_buckets)

@feature(binary=True)
def in_previous_order(previously_ordered):
  return tf.minimum(previously_ordered, 1)

@feature()
def in_previous_order_normalized_cart_order(previously_ordered, n_prev_products):
  return previously_ordered / n_prev_products

if 0:
  @feature(binary=True)
  def days_since_prior_is_maxed(days_since_prior):
    return days_since_prior == 30.0

  @feature(binary=True)
  def sameday(days_since_prior):
    return days_since_prior == 0.0
else:
  days_since_prior_buckets = [2, 3, 5, 7, 10, 15, 20, 30]
  define_bucketized_feature('days_since_prior', days_since_prior_buckets)

# TODO: Probably want to decide on one of either the raw or normalized
# versions of these features
@feature()
def prev_reorder_rate(n_prev_reorders, n_prev_products):
  return n_prev_reorders / n_prev_products
@feature()
def prev_repeat_rate(n_prev_repeats, n_prev_products):
  return n_prev_repeats / n_prev_products

