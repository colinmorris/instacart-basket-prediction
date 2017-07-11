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
from tensorflow.python.client import timeline

import rnnmodel
import utils
import model_helpers
from rnnmodel import RNNModel
from batch_helpers import Batcher


EVAL_PIDS_PER_USER = 2
def evaluate_model(sess, model, batcher):
  total_cost = 0.0
  # The cost measured on just the 'finetuned' metric: the last order
  total_finetune_cost = 0.0
  costvars = [model.cost]
  if not batcher.finetune:
    costvars.append(model.finetune_cost)
  nbatches = 0
  # This seems to work. (Want the pid used for a given user to be the same for each eval run)
  random.seed(1337)
  # TODO: set pids_per_user=-1, and maybe even try smaller final batch
  batches = batcher.get_batches(pids_per_user=EVAL_PIDS_PER_USER, 
      infinite=False,
      allow_smaller_final_batch=False)
  for batch in batches:
    costs = batch_cost(sess, model, batch, 
        train=False, costvars=costvars)
    if batcher.finetune:
      assert len(costs) == 1
      cost = costs[0]
      ft_cost = cost
    else:
      cost, ft_cost = costs
    total_cost += cost
    total_finetune_cost += ft_cost
    nbatches += 1
  return total_cost / nbatches, total_finetune_cost / nbatches

# TODO: yeah, this is a dumb leaky abstraction.
def batch_cost(sess, model, batch, train, costvars=None, 
    rmeta=None, roptions=None, lr=None):
  if costvars is None:
    costvars = [model.cost, model.weight_penalty]
  feed = model_helpers.feed_dict_for_batch(batch, model)
  if train:
    feed[model.lr] = lr
  # TODO: this'll break for a model where product_embeddings is False 
  # (though it seems like I'm probably gonna stick with them)
  to_fetch = costvars
  values = sess.run(to_fetch, feed, options=roptions, run_metadata=rmeta)
  return values
    

def train(sess, model, batcher, runlabel, eval_batcher, eval_model, rmeta, roptions):
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

  log_every = model.hps.log_every
  train_costs = np.zeros(log_every)
  train_ft_costs = np.zeros(log_every)
  l2_costs = np.zeros(log_every)

  batch_fetch_time = 0
  eval_time = 0
  batches = batcher.get_batches(infinite=True)
  # TODO: does evaluating the finetune cost have a non-negligible impact 
  # on training speed?
  # TODO: eventually should probably not bother fetching both cost and
  # finetune_cost when --finetune flag is used. But for now it's kind
  # of nice debugging, to make sure they match.
  train_costvars = [model.cost, model.weight_penalty, model.finetune_cost,
      model.train_op]
  for i in range(hps.num_steps):
    step = sess.run(model.global_step)
    tb0 = time.time()
    batch = batches.next()
    tb1 = time.time()
    batch_fetch_time += (tb1 - tb0)
    lr_exponent = i if model.hps.lr_reset else step
    lr = ( (hps.learning_rate - hps.min_learning_rate) *
           (hps.decay_rate)**lr_exponent + hps.min_learning_rate
         )
    vals = batch_cost(sess, model, batch, 
        train=True, rmeta=rmeta, roptions=roptions, 
        costvars=train_costvars, lr=lr)
    bcost, bl2_cost, b_ft_cost, _ = vals
    costi = i % log_every
    train_costs[costi] = bcost
    l2_costs[costi] = bl2_cost
    train_ft_costs[costi] = b_ft_cost
    if (i+1) % log_every == 0:
      # Average cost over last 100 (or whatever) batches
      cost = train_costs.mean()
      l2_cost = l2_costs.mean()
      ft_cost = train_ft_costs.mean()
      end = time.time()
      time_taken = (end - start) - eval_time

      misc_summ = tf.summary.Summary()
      misc_summ.value.add(tag='Learning_Rate', simple_value=lr)
      cost_summ = tf.summary.Summary()
      cost_summ.value.add(tag='Basic_Train_Cost', simple_value=float(cost))
      cost_summ.value.add(tag='Finetune_Train_Cost', simple_value=float(ft_cost))
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
    if (i+1) % hps.save_every == 0 or i == (hps.num_steps - 1):
      utils.save_model(sess, runlabel, step)
    if (i+1) % hps.eval_every == 0:  
      t0 = time.time()
      eval_cost, ft_cost = evaluate_model(sess, eval_model, eval_batcher)
      t1 = time.time()
      eval_time = t1 - t0
      tf.logging.info('Evaluation loss={:.4f} (took {:.1f}s)'.format(eval_cost, eval_time))
      eval_summ = tf.summary.Summary()
      eval_summ.value.add(tag='Eval_Cost', simple_value=eval_cost)
      eval_summ.value.add(tag='Eval_Finetune_Cost', simple_value=ft_cost)
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
  parser.add_argument('-r', '--resume', metavar='TAG',
      help='Load existing checkpoint with the given tag name and resume training')
  parser.add_argument('--profile', action='store_true')
  parser.add_argument('--finetune', action='store_true')
  args = parser.parse_args()
  # If -f is passed in, we'll load the original base config file, and
  # (potentially) overwrite the '_full' version with new inherited defaults.
  hps = model_helpers.hps_for_tag(args.tag, try_full=(not args.force),
      fallback_to_default=False)

  # Write out the full hps, including the ones inherited from defaults. Because
  # defaults can change over time, and mess us up. This is particularly true
  # of features.
  if not hps.fully_specified:
    #assert not args.resume, "No full hp specification found for {}".format(args.tag)
    hps.fully_specified = True
    full_config_path = 'configs/{}_full.json'.format(args.tag)
    with open(full_config_path, 'w') as f:
      f.write( hps.to_json() )
    print "Wrote full inherited hyperparams to {}".format(full_config_path)
  else:
    assert args.force, "This tag has been run before. Please specify --force"

  tf.logging.info('Building model')
  model = RNNModel(hps)
  tf.logging.info('Loading batcher')
  batcher = Batcher(hps, args.recordfile, 
      in_media_res=args.resume is not None,
      finetune=args.finetune
      )

  eval_hps = model_helpers.copy_hps(hps)
  eval_hps.use_recurrent_dropout = False
  eval_recordfname = 'eval.tfrecords'
  eval_batcher = Batcher(eval_hps, eval_recordfname, finetune=args.finetune)
  eval_model = RNNModel(eval_hps, reuse=True)

  sess = tf.InteractiveSession()

  run_metadata = None
  run_options = None
  if args.profile:
    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

  if args.resume:
    tf.logging.info('Loading saved weights')
    cpkt = 'checkpoints/{}'.format(args.resume)
    utils.load_checkpoint(sess, cpkt)
  else:
    sess.run(tf.global_variables_initializer())
  tf.logging.info('Training')
  # TODO: maybe catch KeyboardInterrupt and save model before bailing? 
  # Could be annoying in some cases.
  t0 = time.time()
  train(sess, model, batcher, args.tag, eval_batcher, eval_model, run_metadata, run_options)
  t1 = time.time()
  tf.logging.info('Completed training in {:.1f}s'.format(t1-t0))

  if args.profile:
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
      f.write(ctf)
    print 'Wrote timeline to timeline.json'

if __name__ == '__main__':
  main()
  #cProfile.run('main()', 'runner.profile')
