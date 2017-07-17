"""Calculates fscore on a given set of users using predictions generated according
to some model's predictions.
"""
import time
import argparse
import numpy as np

from baskets import common
from baskets import predictor as pred
from baskets import rnnmodel
from baskets import evaluator
from baskets.batch_helpers import iterate_wrapped_users

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
  args = parser.parse_args()
  
  t0 = time.time()

  predictors = {}
  if args.baseline:
    predictors['baseline'] = pred.PreviousOrderPredictor()

  for tag in args.tags:
    pmap = common.pdict_for_tag(tag, args.recordfile)
    if args.tp:
      predictors['{}-tp'.format(tag)] = pred.ThresholdPredictor(pmap, args.thresh)
    if args.montecarlo:
      predictors['{}-mc'.format(tag)] = \
          pred.HybridThresholdPredictor(pmap, ntrials=args.mc_trials, save=args.save)

  assert predictors

  user_iterator = iterate_wrapped_users(args.recordfile)
  judge = evaluator.Evaluator(user_iterator)
  # TODO: would be real nice to use tensorboard to look at dist. of
  # probabilities/logits/fscores/precisions stuff like that
  results = judge.evaluate(predictors, limit=args.n_users, save=args.save)
  t1 = time.time()
  print "Finished evaluation in {:.1f}s".format(t1-t0)

  for pname, res in results.iteritems():
    print '{}:'.format(pname)
    df = res.to_df()
    print df.mean()
  
if __name__ == '__main__':
  main()
