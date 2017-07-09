import pickle
import logging
import os

from tensorflow.contrib.training import HParams

import rnnmodel
import utils

def feed_dict_for_batch(batch, model):
  """where batch is a thing returned by a Batcher"""
  x, y, seqlens, lossmask, pids, aids, dids, uids = batch
  feed = {
      model.input_data: x,
      model.labels: y,
      model.sequence_lengths: seqlens,
      model.lossmask: lossmask,
  }
  if model.hps.product_embedding_size:
    feed[model.product_ids] = pids
  if model.hps.aisle_embedding_size:
    feed[model.aisle_ids] = aids
  if model.hps.dept_embedding_size:
    feed[model.dept_ids] = dids
  return feed

class NoConfigException(Exception):
  pass

def hps_for_tag(tag, try_full=True, fallback_to_default=True):
  hps = rnnmodel.get_default_hparams()
  config_path = 'configs/{}.json'.format(tag)
  if try_full:
    full_config_path = 'configs/{}_full.json'.format(tag)
    if os.path.exists(full_config_path):
      with open(full_config_path) as f:
        hps.parse_json(f.read())
      return hps
  if os.path.exists(config_path):
    with open(config_path) as f:
      hps.parse_json(f.read())
  else:
    if fallback_to_default:
      logging.warn('No config file found for tag {}. Using default hps.'.format(tag))
    else:
      raise NoConfigException
  return hps

def copy_hps(hps):
  return HParams(**hps.values())

def pdict_for_tag(tag):
  path = 'pdicts/{}.pickle'.format(tag)
  with open(path) as f:
    return pickle.load(f)
def save_pdict_for_tag(tag, pdict):
  path = 'pdicts/{}.pickle'.format(tag)
  with open(path, 'w') as f:
    pickle.dump(dict(pdict), f)

def load_checkpoint_for_tag(tag, sess):
  cpkt_path = 'checkpoints/{}'.format(tag)
  utils.load_checkpoint(sess, cpkt_path)
