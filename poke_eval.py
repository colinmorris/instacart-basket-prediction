import time
import argparse

import numpy as np
import tensorflow as tf
import pandas as pd

import predictor as pred
import rnnmodel
import utils
import evaluator
from batch_helpers import iterate_wrapped_users, UserWrapper
from insta_pb2 import User

def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path')
  parser.add_argument('-t', '--thresh', default=.2, help='Probability threshold '+
      'for taking a product', type=float)
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
  user_pb = User()
  with open('testuser.pb') as f:
    user_pb.ParseFromString(f.read())
  user = UserWrapper(user_pb)
  judge = evaluator.Evaluator([user])
  predictor = pred.RnnModelPredictor(sess, model, args.thresh, predict_nones=1)
  baseline = pred.PreviousOrderPredictor()
  predictor2 = pred.MonteCarloRnnPredictor(sess, model)

  # TODO: would be real nice to use tensorboard to look at dist. of
  # probabilities/logits/fscores/precisions stuff like that
  t0 = time.time()
  baseline_res, res, res2 = judge.evaluate([baseline, predictor, predictor2])
  t1 = time.time()
  print "Finished evaluation in {:.1f}s".format(t1-t0)
  print "BASELINE:"
  dfb = baseline_res.to_df()
  print dfb.describe()
  print "RNN:"
  df = res.to_df()
  print df.describe()

  print "RNN (monte carlo):"
  print res2.to_df().describe()

if __name__ == '__main__':
  main()
