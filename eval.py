import time
import argparse
import numpy as np

import model_helpers as helpers
import predictor as pred
import rnnmodel
import evaluator
from batch_helpers import iterate_wrapped_users

def main():
  np.set_printoptions(precision=4, threshold=50)
  parser = argparse.ArgumentParser()
  parser.add_argument('tag')
  parser.add_argument('-t', '--thresh', default=.2, help='Probability threshold '+
      'for taking a product (default=.2)', type=float)
  parser.add_argument('--recordfile', default='test.tfrecords', 
      help='tfrecords file with the users to test on (default: test.tfrecords)')
  parser.add_argument('--mc-trials', type=int, default=50,
      help='Number of rounds of monte carlo sim to perform per product/threshold (default:50)')
  parser.add_argument('-n', '--n-users', type=int, 
      help='Limit number of users tested on (default: no limit)')
  #parser.add_argument('-c', '--config', default=None,
  #    help='json file with hypeVrparam overwrites (default: infer from tag)') 
  args = parser.parse_args()
  
  t0 = time.time()


  pmap = helpers.pdict_for_tag(args.tag)
  user_iterator = iterate_wrapped_users(args.recordfile)
  judge = evaluator.Evaluator(user_iterator)
  # TODO: ability to configure which predictors to test via command line args
  tpredictor = pred.ThresholdPredictor(pmap, args.thresh)
  baseline = pred.PreviousOrderPredictor()
  mcpredictor = pred.MonteCarloThresholdPredictor(pmap, ntrials=args.mc_trials)
  predictors = [baseline, tpredictor, mcpredictor]
  
  # TODO: would be real nice to use tensorboard to look at dist. of
  # probabilities/logits/fscores/precisions stuff like that
  results = judge.evaluate(predictors, limit=args.n_users)
  t1 = time.time()
  print "Finished evaluation in {:.1f}s".format(t1-t0)
  
  dfs = [res.to_df() for res in results]
  for (predictor, df) in zip(predictors, dfs):
    print "{}:".format(predictor.__class__.__name__)
    #print df.describe()
    print df.mean()

if __name__ == '__main__':
  main()
