from __future__ import absolute_import
from __future__ import division

#import cProfile

import argparse
import time
import logging
import sys
import os
import random

import numpy as np
import tensorflow as tf
import pandas as pd

import rnnmodel
import utils
from rnnmodel import RNNModel
from batch_helpers import Batcher

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
logger.addHandler(_handler)

def get_next_run_num():
  def tryint(x):
    try:
      return int(x)
    except ValueError:
      return 1
  rundirs = map(tryint, os.listdir('logs'))
  if not rundirs:
    return 1
  return max(rundirs) + 1

def eval_model(sess, model, batcher):
  total_cost = 0.0
  nbatches = 0
  # This seems to work. (Want the pid used for a given user to be the same for each eval run)
  random.seed(1337)
  for batch in batcher.iter_epoch():
    cost = batch_cost(sess, model, batch, train=False)
    total_cost += cost
    nbatches += 1
  return total_cost / nbatches

def batch_cost(sess, model, batch, train):
  x, y, seqlens, lossmask, pids = batch
  feed = {
      model.input_data: x,
      model.labels: y,
      model.sequence_lengths: seqlens,
      model.lossmask: lossmask,
  }
  if model.hps.product_embeddings:
    feed[model.product_ids] = pids
  to_fetch = [model.cost]
  if train:
    to_fetch.append(model.train_op)
  values = sess.run(to_fetch, feed)
  return values[0]
    

def train(sess, model, batcher, runlabel, eval_batcher): # TODO: eval_model
  # Setup summary writer.
  summary_writer = tf.summary.FileWriter('logs/{}'.format(runlabel))

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

  batch_fetch_time = 0
  for i in range(hps.num_steps):
    step = sess.run(model.global_step)
    tb0 = time.time()
    batch = batcher.get_batch(i)
    tb1 = time.time()
    batch_fetch_time += (tb1 - tb0)
    cost = batch_cost(sess, model, batch, train=True)
    if step % 100 == 0 and step > 0 or (hps.num_steps <= 100 and step % 20 == 0 and step > 0):

      end = time.time()
      time_taken = end - start

      cost_summ = tf.summary.Summary()
      cost_summ.value.add(tag='Train_Cost', simple_value=float(cost))
      time_summ = tf.summary.Summary()
      time_summ.value.add(tag='Time_Taken_Train', simple_value=float(time_taken))
      time_summ.value.add(tag='Time_Taken_Batchfetch', simple_value=batch_fetch_time)
      batch_fetch_time = 0

      output_format = ('step: %d, cost: %.4f, train_time_taken: %.4f')
      output_values = (step, cost, time_taken)
      output_log = output_format % output_values
      tf.logging.info(output_log)

      summary_writer.add_summary(cost_summ, step)
      summary_writer.add_summary(time_summ, step)
      summary_writer.flush()
      start = time.time()
    if (step+1) % hps.save_every == 0 or i == (hps.num_steps - 1):
      utils.save_model(sess, runlabel, step)
    if (step+1) % hps.eval_every == 0:  
      t0 = time.time()
      eval_cost = eval_model(sess, model, eval_batcher)
      t1 = time.time()
      eval_time = t1 - t0
      tf.logging.info('Evaluation loss={:.4f} (took {:.1f}s)'.format(eval_cost, eval_time))
      eval_summ = tf.summary.Summary()
      eval_summ.value.add(tag='Eval_Cost', simple_value=eval_cost)
      eval_summ.value.add(tag='Eval_Time', simple_value=eval_time)


def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  # TODO: allow passing in hparams config json
  parser.add_argument('-r', '--run-label', default=None)
  parser.add_argument('-c', '--config', default=None,
      help='json file with hyperparam overwrites') 
  parser.add_argument('--recordfile', default='train.tfrecords', 
      help='tfrecords file with the users to train on (default: train.tfrecords)')
  args = parser.parse_args()
  hps = rnnmodel.get_default_hparams()
  if args.config:
    with open(args.config) as f:
      hps.parse_json(f.read())

  logger.info('Building model')
  model = RNNModel(hps)
  logger.info('Loading batcher')
  batcher = Batcher(hps, args.recordfile)
  eval_recordfname = 'eval.tfrecords'
  eval_batcher = Batcher(hps, eval_recordfname)
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  logger.info('Training')
  if args.run_label is None:
    runlabel = get_next_run_num()
  else:
    runlabel = args.run_label
  # TODO: maybe catch KeyboardInterrupt and save model before bailing? 
  # Could be annoying in some cases.
  train(sess, model, batcher, runlabel, eval_batcher)

if __name__ == '__main__':
  logger.setLevel(logging.INFO)
  main()
  #cProfile.run('main()', 'runner.profile')
