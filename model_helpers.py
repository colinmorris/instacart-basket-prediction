import pickle
import logging
import os

import rnnmodel

def feed_dict_for_batch(batch, model):
  """where batch is a thing returned by a Batcher"""
  x, y, seqlens, lossmask, pids, uids = batch
  feed = {
      model.input_data: x,
      model.labels: y,
      model.sequence_lengths: seqlens,
      model.lossmask: lossmask,
  }
  if model.hps.product_embeddings:
    feed[model.product_ids] = pids
  return feed

def hps_for_tag(tag):
  hps = rnnmodel.get_default_hparams
  config_path = 'configs/{}.json'.format(tag)
  if os.path.exists(config_path):
    with open(config_path) as f:
      hps.parse_json(f.read())
  else:
    logging.warn('No config file found for tag {}. Using default hps.'.format(tag))
  return hps

def pdict_for_tag(tag):
  path = 'pdicts/{}.pickle'.format(tag)
  with open(path) as f:
    return pickle.load(f)
