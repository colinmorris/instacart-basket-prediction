from __future__ import absolute_import
from __future__ import division

#import cProfile

import argparse
import time
import sys
import os
import random

import numpy as np
import tensorflow as tf
import pandas as pd

import rnnmodel
import utils
import model_helpers
from rnnmodel import RNNModel
from batch_helpers import Batcher


EVAL_PIDS_PER_USER = 2
def evaluate_model(sess, model, batcher):
  total_cost = 0.0
  nbatches = 0
  # This seems to work. (Want the pid used for a given user to be the same for each eval run)
  random.seed(1337)
  # TODO: set pids_per_user=-1, and maybe even try smaller final batch
  batches = batcher.get_batches(pids_per_user=EVAL_PIDS_PER_USER, 
      infinite=False,
      allow_smaller_final_batch=False)
  for batch in batches:
    cost, _l2cost = batch_cost(sess, model, batch, train=False)
    total_cost += cost
    nbatches += 1
  return total_cost / nbatches

def batch_cost(sess, model, batch, train, lr=None):
  """Return tuple of basic_cost, regularization_cost"""
  feed = model_helpers.feed_dict_for_batch(batch, model)
  if train:
    feed[model.lr] = lr
  # TODO: this'll break for a model where product_embeddings is False 
  # (though it seems like I'm probably gonna stick with them)
  to_fetch = [model.cost, model.weight_penalty]
  if train:
    to_fetch.append(model.train_op)
  values = sess.run(to_fetch, feed)
  return values[:2]
    

def train(sess, model, batcher, runlabel, eval_batcher, eval_model):
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

  log_every = 500
  train_costs = np.zeros(log_every)
  l2_costs = np.zeros(log_every)

  batch_fetch_time = 0
  eval_time = 0
  batches = batcher.get_batches(infinite=True)
  for i in range(hps.num_steps):
    step = sess.run(model.global_step)
    tb0 = time.time()
    batch = batches.next()
    tb1 = time.time()
    batch_fetch_time += (tb1 - tb0)
    lr = ( (hps.learning_rate - hps.min_learning_rate) *
           (hps.decay_rate)**step + hps.min_learning_rate
         )
    bcost, bl2_cost = batch_cost(sess, model, batch, train=True, lr=lr)
    costi = i % log_every
    train_costs[costi] = bcost
    l2_costs[costi] = bl2_cost
    if (step+1) % log_every == 0:
      # Average cost over last 100 (or whatever) batches
      cost = train_costs.mean()
      l2_cost = l2_costs.mean()
      end = time.time()
      time_taken = (end - start) - eval_time

      misc_summ = tf.summary.Summary()
      misc_summ.value.add(tag='Learning_Rate', simple_value=lr)
      cost_summ = tf.summary.Summary()
      cost_summ.value.add(tag='Basic_Train_Cost', simple_value=float(cost))
      cost_summ.value.add(tag='Weight_Penalty', simple_value=float(l2_cost))
      cost_summ.value.add(tag='Total_Train_Cost', simple_value=float(cost+l2_cost))
      time_summ = tf.summary.Summary()
      time_summ.value.add(tag='Time_Taken_Train', simple_value=float(time_taken))
      time_summ.value.add(tag='Time_Taken_Batchfetch', simple_value=batch_fetch_time)
      batch_fetch_time = 0

      output_format = ('step: %d, cost: %.4f, train_time_taken: %.3f, lr: %.5f')
      output_values = (step, cost, time_taken, lr)
      output_log = output_format % output_values
      tf.logging.info(output_log)

      summary_writer.add_summary(cost_summ, step)
      summary_writer.add_summary(time_summ, step)
      summary_writer.add_summary(misc_summ, step)
      summary_writer.flush()
      start = time.time()
    if (step+1) % hps.save_every == 0 or i == (hps.num_steps - 1):
      utils.save_model(sess, runlabel, step)
    if (step+1) % hps.eval_every == 0:  
      t0 = time.time()
      eval_cost = evaluate_model(sess, eval_model, eval_batcher)
      t1 = time.time()
      eval_time = t1 - t0
      tf.logging.info('Evaluation loss={:.4f} (took {:.1f}s)'.format(eval_cost, eval_time))
      eval_summ = tf.summary.Summary()
      eval_summ.value.add(tag='Eval_Cost', simple_value=eval_cost)
      eval_summ.value.add(tag='Eval_Time', simple_value=eval_time)
      summary_writer.add_summary(eval_summ, step)
      summary_writer.flush()
      eval_time = 0


def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('tag')
  parser.add_argument('--recordfile', default='train.tfrecords', 
      help='tfrecords file with the users to train on (default: train.tfrecords)')
  parser.add_argument('-f', '--force', action='store_true',
      help='If a run with this tag exists, then overwrite it (or maybe it makes a new run side-by-side? I make no guarantees.')
  parser.add_argument('-r', '--resume', action='store_true',
      help='Load existing checkpoint with this tag name and resume training')
  args = parser.parse_args()
  hps = model_helpers.hps_for_tag(args.tag, fallback_to_default=False)

  # Write out the full hps, including the ones inherited from defaults. Because
  # defaults can change over time, and mess us up. This is particularly true
  # of features.
  if not hps.fully_specified:
    assert not (args.force or args.resume), "No full hp specification found for {}".format(args.tag)
    hps.fully_specified = True
    full_config_path = 'configs/{}_full.json'.format(args.tag)
    with open(full_config_path, 'w') as f:
      f.write( hps.to_json() )
    print "Wrote full inherited hyperparams to {}".format(full_config_path)
  else:
    assert args.force or args.resume, "This tag has been run before. Please specify --force or --resume"

  tf.logging.info('Building model')
  model = RNNModel(hps)
  tf.logging.info('Loading batcher')
  # TODO: A model that is resumed a few times will be disproportionately exposed
  # to users at the beginning of the list. Might want to add option to batcher
  # to skip a random number of users on startup.
  batcher = Batcher(hps, args.recordfile)

  eval_hps = model_helpers.copy_hps(hps)
  eval_hps.use_recurrent_dropout = False
  eval_recordfname = 'eval.tfrecords'
  eval_batcher = Batcher(eval_hps, eval_recordfname)
  eval_model = RNNModel(eval_hps, reuse=True)

  sess = tf.InteractiveSession()
  if args.resume:
    tf.logging.info('Loading saved weights')
    cpkt = 'checkpoints/{}'.format(args.tag)
    utils.load_checkpoint(sess, cpkt)
  else:
    sess.run(tf.global_variables_initializer())
  tf.logging.info('Training')
  # TODO: maybe catch KeyboardInterrupt and save model before bailing? 
  # Could be annoying in some cases.
  train(sess, model, batcher, args.tag, eval_batcher, eval_model)

if __name__ == '__main__':
  main()
  #cProfile.run('main()', 'runner.profile')
