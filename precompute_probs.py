#!/usr/bin/env python
import os
import argparse
import tensorflow as tf
import numpy as np
from scipy.special import expit
from collections import defaultdict

from baskets import utils, common, hypers, rnnmodel
from baskets.dataset import BasketDataset
from baskets.time_me import time_me

def get_probmap(model, sess):
  """{uid -> {pid -> prob}}"""
  # Start a fresh pass through the validation data
  sess.run(model.dataset.new_epoch_op())
  pmap = defaultdict(dict)
  i = 0
  nseqs = 0
  to_fetch = [model.lastorder_logits, model.dataset['uid'], model.dataset['pid']]
  while 1:
    try:
      final_logits, uids, pids = sess.run(to_fetch)
    except tf.errors.OutOfRangeError:
      break
    batch_size = len(uids)
    nseqs += batch_size
    final_probs = expit(final_logits)
    for uid, pid, prob in zip(uids, pids, final_probs):
      pmap[uid][pid] = prob
    i += 1
  tf.logging.info("Computed probabilities for {} users over {} sequences in {} batches".format(
    len(pmap), nseqs, i
    ))
  return pmap

def precompute_probs_for_tag(tag, userfold):
  hps = hypers.hps_for_tag(tag, mode=hypers.Mode.inference)
  tf.logging.info('Creating model')
  dat = BasketDataset(hps, userfold)
  model = rnnmodel.RNNModel(hps, dat)
  sess = tf.InteractiveSession()
  # Load pretrained weights
  tf.logging.info('Loading weights')
  utils.load_checkpoint_for_tag(tag, sess)
  # TODO: deal with 'test mode'
  tf.logging.info('Calculating probabilities')
  probmap = get_probmap(model, sess)
  common.save_pdict_for_tag(tag, probmap, userfold)
  sess.close()
  tf.reset_default_graph()
  return probmap

def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('tags', nargs='+')
  parser.add_argument('--fold', default='test.tfrecords', 
      help='fold of users to compute probs for (should correspond to name of a vector file)')
  args = parser.parse_args()

  for tag in args.tags:
    tf.logging.info('Computing probs for tag {}'.format(tag))
    with time_me('Computed probs for {}'.format(tag)):
      precompute_probs_for_tag(tag, args.fold)

if __name__ == '__main__':
  with time_me():
    main()
