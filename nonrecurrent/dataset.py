import xgboost as xgb
import numpy as np
import scipy
from sklearn.preprocessing import label_binarize
import logging

from baskets import common, constants
from baskets.hypers import Mode

import fields

class Dataset(object):
  """Wrapper around the recarrays serialized by scalar_vectorize.py"""

  non_feat_cols = {'label', 'user_prods', 'uid', 'orderid'}
  feat_cols = [col for col in fields.all_fields if col not in non_feat_cols]

  def __init__(self, data_tag, hps, mode=Mode.training, maxlen=None):
    self.hps = hps
    self.mode = mode
    npy_path = common.resolve_scalarvector_path(data_tag)
    dat = np.load(npy_path)
    if maxlen:
      dat = dat[:maxlen]
    self.records = dat
    self.labels = dat['label']

  FIELD_TO_NVALUES = {'pid': constants.N_PRODUCTS, 'aisleid': constants.N_AISLES,
      'deptid': constants.N_DEPARTMENTS}
  def as_dmatrix(self):
    kwargs = dict(label=self.labels)
    # Starts with a dummy value because of reasons.
    onehot_vars = self.hps.onehot_vars[1:]
    if not self.hps.onehot_vars:
      kwargs['feature_names'] = self.feat_cols
    weight =  (self.mode == Mode.eval and self.hps.weight_validation) or\
        (self.mode == Mode.training and self.hps.weight)
    if weight:
      kwargs['weight'] = 1 / self.records['user_prods']

    basic_feat_cols = [col for col in self.feat_cols if col not in onehot_vars]
    featdat = self.records[basic_feat_cols]
    featdat = featdat.view(fields.dtype).reshape(len(featdat), -1)
    onehot_matrices = []
    for onehot_var in onehot_vars:
      onehot = label_binarize(self.records[onehot_var], 
          classes=range(1, self.FIELD_TO_NVALUES[onehot_var]+1),
          sparse_output=True).astype(fields.dtype)
      onehot_matrices.append(onehot)
    if onehot_matrices:
      # TODO: There are some perf issues with this. Look into this workaround:
      # https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices/33259578#33259578
      featdat = scipy.sparse.hstack([featdat,]+onehot_matrices)

    logging.info('Made dmatrix with feature data having shape {}'.format(featdat.shape))

    return xgb.DMatrix(featdat, **kwargs)
