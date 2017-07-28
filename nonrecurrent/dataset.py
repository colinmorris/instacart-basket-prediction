import xgboost as xgb
import numpy as np
import scipy
from sklearn.preprocessing import label_binarize
import logging
import os

from baskets import common, constants
from baskets.hypers import Mode

import fields
import cache_embeddings

# This seems surprisingly unhelpful, but leaving it on for now out of principle.
FTYPES = 1

class Dataset(object):
  """Wrapper around the recarrays serialized by scalar_vectorize.py"""

  CACHE_DIR = os.path.join(common.XGBOOST_DIR, 'cache')
  non_feat_cols = {'label', 'user_prods', 'uid', 'orderid'}
  feat_cols = [col for col in fields.all_fields if col not in non_feat_cols]
  FIELD_TO_NVALUES = {'pid': constants.N_PRODUCTS, 'aisleid': constants.N_AISLES,
      'deptid': constants.N_DEPARTMENTS}

  def __init__(self, data_tag, hps, mode=Mode.training):
    """data_tag is something like 'train', 'train_5', 'validation', etc.
    an identifier for a scalar_vectors file"""
    self.hps = hps
    self.mode = mode
    self.data_tag = data_tag
    self._records = None

  @classmethod
  def basic_feat_cols_for_hps(kls, hps):
    """Simple scalar features (i.e. not one-hot encoded or embedded categoricals)"""
    onehot_vars = kls.onehot_vars_for_hps(hps)
    dropped = hps.dropped_cols[1:]
    return [col for col in kls.feat_cols if 
        col not in onehot_vars and col not in dropped]

  @property
  def basic_feat_cols(self):
    return self.basic_feat_cols_for_hps(self.hps)

  @classmethod
  def feature_names_for_hps(kls, hps):
    onehot_vars = kls.onehot_vars_for_hps(hps)
    cols = kls.basic_feat_cols_for_hps(hps)
    if hps.embedding_tag:
      emb_cols = ['pid_emb_{}'.format(i) for i in range(hps.embedding_dimension)]
      cols += emb_cols
    for onehot_var in onehot_vars:
      onehot_cols = ['{}_{}'.format(onehot_var, i+1) 
          for i in range(kls.FIELD_TO_NVALUES[onehot_var])
          ]
      cols += onehot_cols


    return cols

  @property
  def feature_names(self):
    return self.feature_names_for_hps(self.hps)

  @property
  def feature_types(self):
    types = []
    for fname in self.basic_feat_cols:
      t = 'float' if fname in fields.float_feats else 'int'
      types.append(t)
    if self.hps.embedding_tag:
      emb_types = ['float' for i in range(self.hps.embedding_dimension)]
      types += emb_types
    for onehot_var in self.onehot_vars:
      # 'i' = indicator
      onehot_types = ['i' for _ in range(self.FIELD_TO_NVALUES[onehot_var])]
      types.extend(onehot_types)
    return types

  @classmethod
  def onehot_vars_for_hps(kls, hps):
    return sorted(hps.onehot_vars[1:])

  @property
  def onehot_vars(self):
    return sorted(self.hps.onehot_vars[1:])

  @property
  def weight_mode(self):
    """One of {none, simple, soft}"""
    if self.mode == Mode.eval:
      return 'simple' if self.hps.weight_validation else 'none'
    elif self.mode == Mode.training:
      if not self.hps.weight:
        return 'none'
      else:
        return 'soft' if self.hps.soft_weights else 'simple'
    else:
      assert self.mode == Mode.inference
      return 'none'

  @property
  def dmatrix_key(self):
    k = self.data_tag
    if self.hps.embedding_tag:
      k += '_emb_{}'.format(self.hps.embedding_tag)
    if self.onehot_vars:
      k += '_' + ':'.join(self.onehot_vars)
    return k

  @property
  def dmatrix_cache_path(self):
    fname = '{}.buffer'.format(self.dmatrix_key)
    path = os.path.join(self.CACHE_DIR, fname)
    return path

  def as_dmatrix(self):
    path = self.dmatrix_cache_path
    # xgb is not try/except friendly here
    if os.path.exists(path):
      dm = xgb.DMatrix(path, feature_names=self.feature_names,
          feature_types=(self.feature_types if FTYPES else None)
          )
    else:
      logging.info('Cache miss on dmatrix. Building and caching.')
      dm = self._as_dmatrix()
      dm.save_binary(path)
    # We add on weights (if any) after the fact, to avoid proliferation of big
    # serialized dmatrix files.
    if self.weight_mode != 'none':
      weights = self.get_weights()
      dm.set_weight(weights)
    return dm

  @property
  def records(self):
    if self._records is None:
      npy_path = common.resolve_scalarvector_path(self.data_tag)
      self._records = np.load(npy_path)
    return self._records

  def _as_dmatrix(self):
    kwargs = dict(label=self.records['label'])
    kwargs['feature_names'] = self.feature_names

    featdat = self.records[self.basic_feat_cols]
    featdat = featdat.view(fields.dtype).reshape(len(featdat), -1)

    if self.hps.embedding_tag:
      embs = cache_embeddings.load_embeddings(self.hps.embedding_tag)
      npids, embsize = embs.shape
      assert embsize == self.hps.embedding_dimension
      logging.info('Loaded {}-d embeddings from rnn model {}'.format(
        embsize, self.hps.embedding_tag))
      pids = self.records['pid']
      # NB: pids are 1-indexed
      pidxs = (pids-1).astype(np.int32)
      lookuped = embs[pidxs]
      orig_shape = featdat.shape
      featdat = np.hstack((featdat, lookuped))
      logging.info('Shape went from {} to {} after adding pid embeddings'.format(
        orig_shape, featdat.shape))
    
    onehot_matrices = []
    for onehot_var in self.onehot_vars:
      onehot = label_binarize(self.records[onehot_var], 
          classes=range(1, self.FIELD_TO_NVALUES[onehot_var]+1),
          sparse_output=True).astype(fields.dtype)
      onehot_matrices.append(onehot)
    if onehot_matrices:
      # TODO: There are some perf issues with this. Look into this workaround:
      # https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices/33259578#33259578
      featdat = scipy.sparse.hstack([featdat,]+onehot_matrices)
    
    logging.info('Made dmatrix with feature data having shape {}'.format(featdat.shape))

    # https://github.com/dmlc/xgboost/issues/2554
    if not kwargs['label'].flags.c_contiguous:
      logging.info('Contiguizing labels')
      kwargs['label'] = np.ascontiguousarray(kwargs['label'])
      logging.info('Contiguized')
    if isinstance(featdat, np.ndarray) and not featdat.flags.c_contiguous:
      logging.info('Contiguizing feature data')
      featdat = np.ascontiguousarray(featdat)

    if FTYPES:
      kwargs['feature_types'] = self.feature_types

    return xgb.DMatrix(featdat, **kwargs)

  @property
  def weight_key(self):
    return 'weights_{}_{}'.format(self.data_tag, self.weight_mode)

  def get_weights(self):
    path = os.path.join(self.CACHE_DIR, '{}.npy'.format(self.weight_key))
    try:
      return np.load(path)
    except IOError:
      logging.info('Cache miss on weights with key {}'.format(self.weight_key))
      w = self._get_weights()
      np.save(path, w)
      return w

  def _get_weights(self):
    if self.weight_mode == 'simple':
      return 1 / self.records['user_prods']
    elif self.weight_mode == 'soft':
      return 1 / (np.log2(self.records['user_prods'])+1)
    else:
      assert False, "No weights to get"

  @property
  def uids(self):
    fname = 'uids_{}.npy'.format(self.data_tag)
    uid_path = os.path.join(self.CACHE_DIR, fname)
    try:
      return np.load(uid_path)
    except IOError:
      logging.info('Cache miss on uids')
      uids = self.records['uid'].copy() # contiguize just in case
      np.save(uid_path, uids)
      return uids

  @property
  def pids(self):
    fname = 'pids_{}.npy'.format(self.data_tag)
    pid_path = os.path.join(self.CACHE_DIR, fname)
    try:
      return np.load(pid_path)
    except IOError:
      logging.info('Cache miss on pids')
      pids = self.records['pid'].copy() # contiguize just in case
      np.save(pid_path, pids)
      return pids
