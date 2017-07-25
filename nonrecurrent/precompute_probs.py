#!/usr/bin/env python

import argparse
import logging
import xgboost as xgb
from collections import defaultdict

from baskets import common
from baskets.time_me import time_me

from dataset import Dataset

def get_pdict(model, dataset):
  """{uid -> {pid -> prob}}"""
  dtest = dataset.as_dmatrix()
  probs = model.predict(dtest)
  pmap = defaultdict(dict)
  for i, record in enumerate(dataset.records):
    uid, pid = map(int, [record['uid'], record['pid']])
    pmap[uid][pid] = probs[i]
  return pmap

def main():
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('tags', metavar='tag', nargs='+')
  parser.add_argument('--recordfile', default='test', 
      help='identifier for file with the users to test on (default: test)')
  parser.add_argument('-n', '--n-users', type=int, 
      help='Limit number of users tested on (default: no limit)')
  args = parser.parse_args()

  dataset = Dataset(args.recordfile, maxlen=args.n_users)
  for model_tag in args.tags:
    path = common.resolve_xgboostmodel_path(model_tag)
    logging.info('Loading model with tag {}'.format(model_tag))
    model = xgb.Booster(model_file=path)
    logging.info('Computing probs for tag {}'.format(model_tag))
    with time_me('Computed probs for {}'.format(model_tag), mode='stderr'):
      pdict = get_pdict(model, dataset)
      logging.info('Got probs for {} users'.format(len(pdict)))
      # TODO: might want to enforce some namespace separation between 
      # rnn-generated pdicts and ones coming from xgboost models?
      common.save_pdict_for_tag(model_tag, pdict, args.recordfile) 

if __name__ == '__main__':
  main()
