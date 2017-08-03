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
  to_fetch = [model.lastorder_logits, model.dataset['uid'], model.dataset['pid']]
  while 1:
    try:
      final_logits, uids, pids = sess.run(to_fetch)
    except tf.errors.OutOfRangeError:
      break
    final_probs = expit(final_logits)
    for uid, pid, prob in zip(uids, pids, final_probs):
      pmap[uid][pid] = prob
    i += 1
  tf.logging.info("Computed probabilities for {} users in {} batches".format(len(pmap), i))
  return pmap

def precompute_probs_for_tag(tag, args):
  hps = hypers.hps_for_tag(tag, mode=hypers.Mode.inference)
  tf.logging.info('Creating model')
  dat = BasketDataset(hps, args.recordfile)
  model = rnnmodel.RNNModel(hps, dat)
  sess = tf.InteractiveSession()
  # Load pretrained weights
  tf.logging.info('Loading weights')
  utils.load_checkpoint_for_tag(tag, sess)
  # TODO: deal with 'test mode'
  tf.logging.info('Calculating probabilities')
  probmap = get_probmap(model, sess)
  common.save_pdict_for_tag(tag, probmap, args.recordfile)
  sess.close()
  tf.reset_default_graph()

def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('tags', metavar='tag', nargs='+')
  parser.add_argument('--recordfile', default='test.tfrecords', 
      help='tfrecords file with the users to test on (default: test.tfrecords)')
  args = parser.parse_args()

  for tag in args.tags:
    tf.logging.info('Computing probs for tag {}'.format(tag))
    with time_me('Computed probs for {}'.format(tag)):
      precompute_probs_for_tag(tag, args)

if __name__ == '__main__':
  with time_me():
    main()
