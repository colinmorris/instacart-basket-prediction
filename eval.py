#!/usr/bin/env python
"""Calculates fscore on a given set of users using predictions generated according
to some model's predictions.
"""
import argparse
import logging

from baskets import common
from baskets import predictor as pred
from baskets import rnnmodel
from baskets import evaluator
from baskets.user_wrapper import iterate_wrapped_users
from baskets.time_me import time_me

import precompute_probs

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('tags', metavar='tag', nargs='+')
  parser.add_argument('-t', '--thresh', default=.2, help='Probability threshold '+
      'for taking a product when using threshold predictor (default=.2)', type=float)
  parser.add_argument('--recordfile', default='test.tfrecords', 
      help='tfrecords file with the users to test on (default: test.tfrecords)')
  parser.add_argument('--mc-trials', type=int, default=50,
      help='Number of rounds of monte carlo sim to perform per product/threshold (default:50)')
  parser.add_argument('-n', '--n-users', type=int, 
      help='Limit number of users tested on (default: no limit)')
  parser.add_argument('--baseline', action='store_true', 
      help='Run a dumb baseline predict-previous-order predictor for comparison')
  parser.add_argument('--tp', action='store_true', 
      help='Run a basic thresholded predictor for each tag using --thresh threshold')
  parser.add_argument('--mc', action='store_true', dest='montecarlo', default=True,
      help='Run a monte-carlo thresh predictor per tag')
  parser.add_argument('--no-mc', action='store_false', dest='montecarlo',
      help='Don\'t run a monte-carlo thresh predictor per tag')
  parser.add_argument('--save', action='store_true', help='Serialize predictions to a file')
  parser.add_argument('--quick', action='store_true', help='Cut some corners')
  args = parser.parse_args()
  
  predictors = {}
  if args.baseline:
    predictors['baseline'] = pred.PreviousOrderPredictor()

  for tag in args.tags:
    try:
      pmap = common.pdict_for_tag(tag, args.recordfile)
    except common.NoPdictException as err:
      logging.warning(err.message + "\nPrecomputing and saving probabilities")
      # Not clear whether this 'recovery' mode should be on by default. Might cause more problems than it solves.
      # Not every tag belongs to an rnn model.
      with time_me('Precomputed probabilities', mode='stderr'):
        pmap = precompute_probs.precompute_probs_for_tag(tag, args.recordfile)
    if args.tp:
      predictors['{}-tp'.format(tag)] = pred.ThresholdPredictor(pmap, args.thresh)
    if args.montecarlo:
      predictors['{}-mc'.format(tag)] = \
          pred.HybridThresholdPredictor(pmap, 
              ntrials=args.mc_trials, 
              save=args.save,
              optimization_level=(0 if args.quick else 10)
              )

  assert predictors

  user_iterator = iterate_wrapped_users(args.recordfile)
  judge = evaluator.Evaluator(user_iterator)
  # TODO: would be real nice to use tensorboard to look at dist. of
  # probabilities/logits/fscores/precisions stuff like that
  results = judge.evaluate(predictors, limit=args.n_users, save=args.save)

  for pname, res in results.iteritems():
    print '{}:'.format(pname)
    df = res.to_df()
    print df.mean()
  
if __name__ == '__main__':
  with time_me('Finished evaluation', mode='stderr'):
    main()
