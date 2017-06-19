from __future__ import absolute_import

import argparse
import time
import logging
import sys
import os

import numpy as np
import tensorflow as tf
import pandas as pd

import rnnmodel
from rnnmodel import RNNModel
from dataset import Dataset

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
logger.addHandler(_handler)

def get_next_run_num():
  rundirs = map(int, os.listdir('logs'))
  if not rundirs:
    return 1
  return max(rundirs) + 1  


def train(sess, model, data): # TODO: eval_model
  # Setup summary writer.
  summary_writer = tf.summary.FileWriter('logs/{}'.format(
      get_next_run_num()
    ))

  # Calculate trainable params.
  t_vars = tf.trainable_variables()
  count_t_vars = 0
  for var in t_vars:
    num_param = np.prod(var.get_shape().as_list())
    count_t_vars += num_param
    tf.logging.info('%s %s %i', var.name, str(var.get_shape()), num_param)
  tf.logging.info('Total trainable variables %i.', count_t_vars)
  model_summ = tf.summary.Summary()
  model_summ.value.add(
      tag='Num_Trainable_Params', simple_value=float(count_t_vars))
  summary_writer.add_summary(model_summ, 0)
  summary_writer.flush()
  
  hps = model.hps
  start = time.time()

  for i in range(hps.num_steps):
    step = sess.run(model.global_step)
    y, x, seqlens = data.get_batch(i)
    feed = {
        model.input_data: x,
        model.labels: y,
        model.sequence_lengths: seqlens,
    }
    cost, _ = sess.run([model.cost, model.train_op], feed)
    if step % 20 == 0 and step > 0:

      end = time.time()
      time_taken = end - start

      cost_summ = tf.summary.Summary()
      cost_summ.value.add(tag='Train_Cost', simple_value=float(cost))
      time_summ = tf.summary.Summary()
      time_summ.value.add(
          tag='Time_Taken_Train', simple_value=float(time_taken))

      output_format = ('step: %d, cost: %.4f, train_time_taken: %.4f')
      output_values = (step, cost, time_taken)
      output_log = output_format % output_values
      tf.logging.info(output_log)

      summary_writer.add_summary(cost_summ, step)
      summary_writer.add_summary(time_summ, step)
      summary_writer.flush()
      start = time.time()
    # TODO: save_every stuff

def main():
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  #hps = rnnmodel.get_default_hparams()
  hps = rnnmodel.get_toy_hparams()
  logger.info('Building model')
  model = RNNModel(hps)
  logger.info('Loading dataset')
  data = Dataset.load(hps)
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  logger.info('Training')
  train(sess, model, data)

if __name__ == '__main__':
  logger.setLevel(logging.INFO)
  main()
