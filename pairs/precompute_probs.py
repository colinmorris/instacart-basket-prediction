#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from collections import defaultdict

from baskets import common, user_wrapper
from baskets.time_me import time_me

import train
import vectorize

def pdictify(probs, fold):
  pdict = defaultdict(dict)
  i = 0
  for uid, pid in user_wrapper.canonical_ordered_uid_pids(fold):
    p = probs[i]
    pdict[uid][pid] = p
    i += 1
  return pdict

def load_metavectors(fold):
  df = pd.read_pickle('metavectors.pickle')
  df = df[df['fold']==fold]
  return df

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--tag', default='pairs')
  parser.add_argument('--fold', default='test')
  args = parser.parse_args()

  #metavec = load_metavectors(args.fold)

  clf = train.load_model(args.tag)
  X, y = vectorize.load_fold(args.fold, args.tag)

  if hasattr(clf, 'predict_proba'):
    probs = clf.predict_proba(X)
    # returns an array of shape (n, 2), where each len-2 subarray
    # has the probability of the negative and positive classes. which is silly.
    probs = probs[:,1]
  else:
    scores = clf.decision_function(X)
    probs = expit(scores)

  pdict = pdictify(probs, args.fold)
  common.save_pdict_for_tag(args.tag, pdict, args.fold)

if __name__ == '__main__':
  with time_me(mode='print'):
    main()
