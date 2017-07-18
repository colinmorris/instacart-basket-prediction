import tensorflow as tf

from baskets.features import ALL_FEATURES, FEAT_LOOKUP

# Abstraction around a list of features
class FeatureSpec(object):

  def __init__(self, feats):
    self.features = feats

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
  def for_hps(kls, hps):
    feats = []
    for featname in hps.features:
      feat = FEAT_LOOKUP[featname]
      feats.append(feat)
    return FeatureSpec(feats)

  def features_tensor_for_dataset(self, dataset):
    feat_tensors = []
    for feat in self.features:
      feat_tensors.append( feat.fn(dataset) )
    feat_tensor = tf.concat(feat_tensors, axis=0)
    # Above has shape [total_feat_arity, seqlen], so transpose
    return tf.transpose(feat_tensor)

