#!/usr/bin/env python
import argparse
import logging
import numpy as np
import scipy.sparse
import sklearn.svm
import sklearn.linear_model

from baskets import common
from baskets.time_me import time_me

def main():
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser()
  #parser.add_argument('tag')
  args = parser.parse_args()

  X = scipy.sparse.load_npz('vectors.npz')
  y = np.load('labels.npy')

  # Solvers to try:
  # svm.LinearSVC
  # Lasso (sparse coefs, coordinate descent)
  # ElasticNet? (mix of L1 and L2 r12n)
  # Lars (least angle regression - numerically efficient when nfeats >> ninstances)
  # LassoLars (lars + L1 r12n)
  # LogisticRegression: umbrella with a bunch of solvers. liblinear is the only
  #     one that supports l1 r12n. It uses coordinate descent.
  # SGDClassifier (this + log loss recommended for 'large datasets')
  clf = sklearn.svm.LinearSVC(
      C=1.0,
      penalty='l1',
      dual=False,
      verbose=1,
      )

  clf.fit(X, y)

  return clf

if __name__ == '__main__':
  with time_me(mode='print'):
    clf = main()
