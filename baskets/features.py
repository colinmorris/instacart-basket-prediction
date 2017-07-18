import tensorflow as tf
from collections import namedtuple

_raw_feats = ['previously_ordered', 'days_since_prior', 'dow', 'hour',
      'n_prev_products', ]
      #'n_prev_repeats', 'n_prev_reorders']

Feature = namedtuple('Feature', 'name arity fn')

ALL_FEATURES = []
FEAT_LOOKUP = {}

def _normalize_feat_output(output_tensor):
  rank = len(output_tensor.shape)
  if rank == 1:
    output_tensor = tf.expand_dims(output_tensor, 0)
  rank = len(output_tensor.shape)
  assert rank == 2, "Expected rank 2, got {}".format(rank)
  if output_tensor.dtype != tf.float32:
    output_tensor = tf.cast(output_tensor, tf.float32)
  return output_tensor

def feature(arity=1, name=None):
  def decorator(featfn):
    argnames = featfn.func_code.co_varnames
    def wrapped(dataset):
      args = [dataset[arg] for arg in argnames]
      res = featfn(*args)
      return _normalize_feat_output(res)

    featname = name or featfn.__name__
    feat = Feature(featname, arity, wrapped)
    ALL_FEATURES.append(feat)
    assert featname not in FEAT_LOOKUP
    FEAT_LOOKUP[featname] = feat
    return wrapped

  return decorator

def define_passthrough_feature(key):
  fn = lambda ds: _normalize_feat_output(ds[key])
  feat = Feature(key, 1, fn)
  ALL_FEATURES.append(feat)
  FEAT_LOOKUP[key] = feat

_passthrough_keys = ['days_since_prior', 'n_prev_products']
for k in _passthrough_keys:
  define_passthrough_feature(k)

@feature
def in_previous_order(previously_ordered):
  return tf.min(previously_ordered, 1)

@feature(7)
def dow_onehot(dow):
  return tf.one_hot(dow, 7, axis=0)

# TODO: Mooooore features.
