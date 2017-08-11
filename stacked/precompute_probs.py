#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.externals import joblib
from collections import defaultdict

from baskets import common
from baskets.time_me import time_me
import baskets.time_me
baskets.time_me.set_default_mode('print')

import metavectorize

def pdictify(probs, metavec):
  pdict = defaultdict(dict)
  for (i, uid, pid) in metavec[ ['uid', 'pid'] ].itertuples():
    p = probs[i]
    pdict[uid][pid] = p
  return pdict

def load_metavectors(fold):
  df = pd.read_pickle('metavectors.pickle')
  df = df[df['fold']==fold]
  return df

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('tags', nargs='+')
  parser.add_argument('--dest-tag', default='stacked', 
      help='Tag for generated pdict (default: "stacked")')
  parser.add_argument('--fold', default='test')
  args = parser.parse_args()

  metavec = load_metavectors(args.fold)

  #clf = train.load_model()
  clf = joblib.load('model.pkl')

  with time_me('Vectorized fold {}'.format(args.fold)):
    X, y = metavectorize.vectorize_fold(args.fold, args.tags, metavec)

  if hasattr(clf, 'predict_proba'):
    probs = clf.predict_proba(X)
    # returns an array of shape (n, 2), where each len-2 subarray
    # has the probability of the negative and positive classes. which is silly.
    probs = probs[:,1]
  else:
    scores = clf.decision_function(X)
    probs = expit(scores)

  pdict = pdictify(probs, metavec)
  common.save_pdict_for_tag(args.dest_tag, pdict, args.fold)

if __name__ == '__main__':
  with time_me(mode='print'):
    main()
