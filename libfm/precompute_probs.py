#!/usr/bin/env python
from __future__ import division
import argparse
import pickle
from collections import defaultdict

from baskets import common
from baskets.time_me import time_me

def make_pdict(test_ids, probs):
  pdict = defaultdict(dict)
  for (uid, pid), probline in zip(test_ids, probs):
    prob = float(probline.strip())
    pdict[uid][pid] = prob
  return pdict

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--predictions', default='predictions.out')
  parser.add_argument('--tag', default='libfm')
  parser.add_argument('--fold', default='test', help='User fold these examples' +\
      ' came from. Should match whatever was used in vectorize.py (default: test)'
      )
  args = parser.parse_args()

  with open('test_ids.pickle') as f:
    ids = pickle.load(f)
  with open(args.predictions) as probs:
    pdict = make_pdict(ids, probs)

  print 'Made pdict covering {} users'.format(len(pdict))
  common.save_pdict_for_tag(args.tag, pdict, args.fold)

if __name__ == '__main__':
  with time_me(mode='print'):
    main()
