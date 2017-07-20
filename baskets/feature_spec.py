from __future__ import division
import tensorflow as tf
import json
import os

from baskets.features import ALL_FEATURES, FEAT_LOOKUP
from baskets import common

# Abstraction around a list of features
class FeatureSpec(object):

  def __init__(self, feats, normalize=False):
    self.features = feats
    self.normalize = normalize
    self._stats_lookup = None

  @property
  def shape(self):
    total_arity = sum([feat.arity for feat in self.features])
    # first dimension, sequence length, is variable
    return (-1, total_arity)

  @property
  def names(self):
    return [feat.name for feat in self.features]

  @classmethod
  def default_spec(kls):
    return FeatureSpec(ALL_FEATURES)

  @classmethod
  def all_features_spec(kls):
    return FeatureSpec(ALL_FEATURES)

  def features_like_shape(self):
    for feat in self.features:
      for _ in range(feat.arity):
        yield feat

  @classmethod
  def for_hps(kls, hps):
    feats = []
    for featname in hps.features:
      feat = FEAT_LOOKUP[featname]
      feats.append(feat)
    return FeatureSpec(feats, hps.normalize_features)

  @property
  def feature_stats(self):
    if self._stats_lookup is None:
      with open(os.path.join(common.DATA_DIR, 'feature_stats.json')) as f:
        self._stats_lookup = json.load(f)
    return self._stats_lookup

  def _maybe_normalize(self, feat, output_tensor):
    if not self.normalize or feat.binary:
      return output_tensor
    stats = self.feature_stats[feat.name]
    means = [statum['mean'] for statum in stats]
    variances = [statum['variance'] for statum in stats]
    return (output_tensor - means) / variances

  def features_tensor_for_dataset(self, dataset):
    feat_tensors = []
    for feat in self.features:
      tensor = self._maybe_normalize(feat, feat.fn(dataset))
      feat_tensors.append(tensor)
    feat_tensor = tf.concat(feat_tensors, axis=0)
    # Above has shape [total_feat_arity, seqlen], so transpose
    return tf.transpose(feat_tensor)

