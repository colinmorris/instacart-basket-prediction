import time
import argparse

import numpy as np
import tensorflow as tf
import pandas as pd

import predictor as pred
import rnnmodel
import utils
import evaluator
from batch_helpers import iterate_wrapped_users

def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path')
  parser.add_argument('-t', '--thresh', default=.5, help='Probability threshold '+
      'for taking a product', type=float)
  parser.add_argument('--recordfile', default='test.tfrecords', 
      help='tfrecords file with the users to test on (default: test.tfrecords)')
  parser.add_argument('-n', '--n-users', type=int, help='Limit number of users tested on')
  parser.add_argument('-c', '--config', default=None,
      help='json file with hyperparam overwrites') 
  args = parser.parse_args()
  hps = rnnmodel.get_toy_hparams()
  if args.config:
    with open(args.config) as f:
      hps.parse_json(f.read())
  hps.is_training = False
  hps.batch_size = 1
  tf.logging.info('Creating model')
  model = rnnmodel.RNNModel(hps)
  
  sess = tf.InteractiveSession()
  # Load pretrained weights
  tf.logging.info('Loading weights')
  utils.load_checkpoint(sess, args.checkpoint_path)

  tf.logging.info('Loading test set')
  user_iterator = iterate_wrapped_users(args.recordfile)
  judge = evaluator.Evaluator(user_iterator)
  predictor = pred.RnnModelPredictor(sess, model, args.thresh, predict_nones=1)
  baseline = pred.PreviousOrderPredictor()

  # TODO: would be real nice to use tensorboard to look at dist. of
  # probabilities/logits/fscores/precisions stuff like that
  t0 = time.time()
  baseline_res, res = judge.evaluate([baseline, predictor], limit=args.n_users)
  t1 = time.time()
  print "Finished evaluation in {:.1f}s".format(t1-t0)
  print "BASELINE:"
  dfb = baseline_res.to_df()
  print dfb.describe()
  print "RNN:"
  df = res.to_df()
  print df.describe()

if __name__ == '__main__':
  main()
