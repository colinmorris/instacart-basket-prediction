from tensorflow.contrib.training import HParams

from baskets import common
from baskets.hypers import Mode

def get_default_hparams():
  return HParams(
      mode=Mode.training,
      train_file='train', # i.e. train_full
      rounds=50,
      early_stopping_rounds=10,
      # Whether to apply per-instance weighting to the training set
      weight=False,
      weight_validation=True,
      # Ugh. More riggery and tomfoolery.
      # For some reason tf Hparams can't have empty sequence values? But if we default
      # it to None and try to overwrite it with a list later, tf complains "hey, that
      # thing isn't supposed to be a list!". So start it with a dummy value. Bleh.
      onehot_vars=[None],
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
    raise NoHpsDefinedException()
  return hps

def save_hps(tag, hps):
  # (IntEnum will actually pass a check of isinstance int)
  if type(hps.mode) != int:
    hps.mode = hps.mode.value
  path = common.resolve_xgboost_config_path(tag)
  with open(path, 'w') as f:
    f.write(hps.to_json())
