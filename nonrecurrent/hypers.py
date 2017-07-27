import json
from tensorflow.contrib.training import HParams

from baskets import common
from baskets.hypers import Mode

def get_default_hparams():
  return HParams(
      # (not actually used?)
      mode=Mode.training,
      train_file='train', # i.e. train_full
      rounds=50,
      early_stopping_rounds=10,
      # Whether to apply per-instance weighting to the training set
      weight=False,
      # An intermediate option between no weighting and weighting instances
      # by 1/userprods. As a user's number of products increases, the weight
      # on each of their instances increases, but the total weight of all 
      # their instances increases. Only used for training.
      # (If weight is False, the value of soft_weights has no significance)
      soft_weights=False,
      weight_validation=True,
      # Ugh. More riggery and tomfoolery.
      # For some reason tf Hparams can't have empty sequence values? But if we default
      # it to None and try to overwrite it with a list later, tf complains "hey, that
      # thing isn't supposed to be a list!". So start it with a dummy value. Bleh.
      onehot_vars=[None],
      # XXX: Not currently working (haven't had need to use this yet, so it rotted)
      dropped_cols=[None],

      # --- xgb params below --
      # step size shrinkage aka learning_rate
      eta=0.1,
      # Min loss reduction required to make a furth partition on a leaf node.
      gamma=0.70,
      # One of the below two params must be > 0
      max_depth=6,
      # Only relevant if grow_policy == 'lossguide'
      max_leaves=0,
      # Minimum sum of instance weight (hessian) needed in a child. Whatever that means.
      min_child_weight=10,
      max_delta_step=0,
      # Subsample ratio of training data
      subsample=0.76,
      # Sample this fraction of cols per tree
      colsample_bytree=0.65,
      colsample_bylevel = .5,
      # L2 regularization
      reg_lambda=10,
      # L1 regularization
      alpha=2e-05,
      tree_method='auto',
      scale_pos_weight=1,
      grow_policy='depthwise',
      base_score=.2, # (default .5. gives a very minor headstart.)
      max_bins=256,
      )




class NoHpsDefinedException(Exception):
  pass

def hps_for_tag(tag):
  """mode is an optional override for whatever's defined in the config file
  for the mode field"""
  hps = get_default_hparams()
  config_path = common.resolve_xgboost_config_path(tag)
  try:
    with open(config_path) as f:
      hps.parse_json(f.read())
  except IOError:
    msg = "No hps defined for tag {}".format(tag)
    raise NoHpsDefinedException(msg)
  return hps

def save_hps(tag, hps, pprint=True):
  # (IntEnum will actually pass a check of isinstance int)
  hps_dict = hps.values()
  if type(hps.mode) != int:
    hps_dict['mode'] = hps.mode.value
  path = common.resolve_xgboost_config_path(tag)
  with open(path, 'w') as f:
    json.dump(hps_dict, f, indent=2, separators=(',', ': '))
  
_XGB_HPS = ['eta', 'max_depth', 'min_child_weight', 'gamma', 'subsample',
    'colsample_bytree', 'alpha', 'reg_lambda',
    'max_delta_step', 'colsample_bylevel', 'tree_method', 'scale_pos_weight',
    'base_score', 'grow_policy', 'max_leaves', 'max_bins',
]
def xgb_params_from_hps(hps):
  # Copied from https://www.kaggle.com/nickycan/lb-0-3805009-python-edition
  params = dict(objective="reg:logistic", eval_metric="logloss")
  for param in _XGB_HPS:
    params[param] = getattr(hps, param)
  return params
