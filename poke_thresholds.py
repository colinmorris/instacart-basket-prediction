import time
import argparse
import pickle
import numpy as np
import pandas as pd

import model_helpers as helpers
import predictor as pred
from batch_helpers import iterate_wrapped_users
import fscore as fscore_helpers

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('tags', metavar='tag', nargs='+')
  parser.add_argument('--recordfile', default='test.tfrecords', 
      help='tfrecords file with the users to test on (default: test.tfrecords)')
  parser.add_argument('--mc-trials', type=int, default=800,
      help='Number of rounds of monte carlo sim to perform per product/threshold (default:800)')
  parser.add_argument('--exhaustive', action='store_true',
      help='Exhaustively search candidate thresholds')
  parser.add_argument('-n', '--n-users', type=int, 
      help='Limit number of users tested on (default: no limit)')
  args = parser.parse_args()
  
  t0 = time.time()

  assert len(args.tags) == 1
  tag = args.tags[0]
  print "Loading pdict"
  pmap = helpers.pdict_for_tag(tag)
  predictor = pred.HybridThresholdPredictor(pmap, ntrials=args.mc_trials)
  user_iterator = iterate_wrapped_users(args.recordfile)


  dfs = []
  print "Crunching data"
  for i, user in enumerate(user_iterator):
    if args.n_users and i >= args.n_users:
      break
    user_dat = []
    actual_set = user.last_order_predictable_prods()
    pid_to_prob = pmap[user.uid]
    items = pid_to_prob.items()
    # Sort on probability
    items.sort(key = lambda i: i[1])
    pids = [i[0] for i in items]
    probs = [i[1] for i in items]
    probs = np.array(probs)
    p_none = np.product(1-probs)
    actual_arr = np.zeros(len(pids))
    for i, pid in enumerate(pids):
      if pid in actual_set:
        actual_arr[i] = 1
    
    cands = predictor.get_candidate_thresholds(probs, exhaustive=args.exhaustive)
    for cand in cands:
      if args.exhaustive:
        thresh, was_cand = cand
      else:
        thresh = cand
      e_f = predictor.evaluate_threshold(thresh, probs)
      predicted = (probs >= thresh).astype(np.int8)
      predict_none = p_none > thresh or predicted.sum() == 0
      actual_fs = fscore_helpers.fscore(predicted, actual_arr, predict_none)
      row = [thresh, e_f, actual_fs]
      if args.exhaustive:
        row.append(was_cand)
      user_dat.append(row)

    cols = ['thresh', 'e_f', 'fscore']
    if args.exhaustive:
      cols.append('cand')
    df = pd.DataFrame(user_dat, columns=cols)
    dfs.append(df)

  print "Saving"
  with open('threshdat.pickle', 'w') as f:
    pickle.dump(dfs, f)
  
  t1 = time.time()
  print "Finished poking in {:.1f}s".format(t1-t0)
  return dfs
  
if __name__ == '__main__':
  dfs = main()
