import csv
import os
import pickle
import argparse

from baskets import common
from baskets.time_me import time_me

import hypers
from hypers import _XGB_HPS

NONVARIED_XGB_PARAMS = {'base_score', 'scale_pos_weight'}

VARIED_XGB_PARAMS = [param for param in _XGB_HPS if param not in NONVARIED_XGB_PARAMS]

VARIED_HPS = VARIED_XGB_PARAMS + ['weight', 'soft_weights', 'onehot_vars', 'train_file',] 

cols = [
  'tag', 'fscore', 'validation_logloss', 'train_logloss', 'rounds', 'time',
  ] + VARIED_HPS

def get_completed_tags(prefix):
  for fname in os.listdir('results'):
    if prefix and not fname.startswith(prefix):
      continue
    ext = '.pickle'
    assert fname.endswith(ext)
    tag = fname[:-len(ext)]
    yield tag
      
def get_results(tag):
  path = os.path.join(common.XGBOOST_DIR, 'results', tag+'.pickle')
  with open(path) as f:
    return pickle.load(f)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--prefix')
  args = parser.parse_args()
  f = open('results.csv', 'w')
  writer = csv.DictWriter(f, fieldnames=cols)
  writer.writeheader()

  i = 0
  for tag in get_completed_tags(args.prefix):
    hps = hypers.hps_for_tag(tag)
    res = get_results(tag)
    train_losses = res['evals']['train']['logloss']
    val_losses = res['evals']['validation']['logloss']
    row = dict(tag=tag, fscore=res['fscore'], time=res['duration'],
        validation_logloss=min(val_losses), train_logloss=min(train_losses),
        rounds=len(train_losses),
        )
    for hp in VARIED_HPS:
      row[hp] = getattr(hps, hp)
    for (k, v) in row.iteritems():
      if isinstance(v, float):
        if k == 'alpha': # these can be teensy tiny
          fmt = '{:.1e}'
        else:
          fmt = '{:.4f}'
        row[k] = fmt.format(v)
    row['onehot_vars'] = len(hps.onehot_vars)-1
    writer.writerow(row)
    i += 1
  print "Wrote results for {} runs".format(i)

if __name__ == '__main__':
  with time_me(mode='print'):
    main()
