import xgboost as xgb
import numpy as np

from baskets import common

from scalar_vectorize import all_fields

class Dataset(object):
  """Wrapper around the recarrays serialized by scalar_vectorize.py"""

  non_feat_cols = {'label', 'weight', 'uid'}
  feat_cols = [col for col in all_fields if col not in non_feat_cols]

  def __init__(self, tag, maxlen=None):
    npy_path = common.resolve_scalarvector_path(tag)
    dat = np.load(npy_path)
    if maxlen:
      dat = dat[:maxlen]
    self.records = dat
    featdat = dat[self.feat_cols]
    self.featdat = featdat.view(float).reshape(len(featdat), -1)
    self.labels = dat['label']

  def as_dmatrix(self):
    return xgb.DMatrix(self.featdat, label=self.labels, feature_names=self.feat_cols)
