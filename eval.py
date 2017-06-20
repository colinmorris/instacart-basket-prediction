
import argparse

import numpy as np
import tensorflow as tf
import pandas as pd

import predictor as pred
import rnnmodel
import utils
import evaluator
import dataset
from dataset import Dataset

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path')
  parser.add_argument('-t', '--thresh', default=.5, help='Probability threshold '+
      'for taking a product', type=float)
  args = parser.parse_args()
  hps = rnnmodel.get_toy_hparams()
  hps.is_training = False
  hps.batch_size = 1
  tf.logging.info('Creating model')
  model = rnnmodel.RNNModel(hps)
  
  sess = tf.InteractiveSession()
  # Load pretrained weights
  tf.logging.info('Loading weights')
  utils.load_checkpoint(sess, args.checkpoint_path)

  tf.logging.info('Loading test set')
  test = pd.read_pickle(Dataset.TEST_PATH)
  judge = evaluator.Evaluator(test)
  predictor = pred.RnnModelPredictor(sess, model, args.thresh)

  # TODO: would be real nice to use tensorboard to look at dist. of
  # probabilities/logits/fscores/precisions stuff like that
  tf.logging.info('Evaluating predictions on test set with {} rows'.format(
      len(test)))
  res = judge.evaluate(predictor)
  df = res.to_df()
  print df.describe()

if __name__ == '__main__':
  main()
