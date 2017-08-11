#!/usr/bin/env python
import argparse
import logging
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.svm
import sklearn.linear_model
from sklearn.externals import joblib

from baskets import common
from baskets.time_me import time_me

import vectorize

model_fname = 'model.pkl'
def save_model(model, tag=None):
  joblib.dump(model, model_fname)

def load_model(tag=None):
  return joblib.load(model_fname)


def main():
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--train-fold', default='train')
  parser.add_argument('--validation-fold', default='validation')
  args = parser.parse_args()

  X, y = vectorize.load_fold(args.train_fold)

  # Solvers to try:
  # svm.LinearSVC (nb: doesn't do probability estimates. but I guess can use decision function as proxy)
  # Lasso (sparse coefs, coordinate descent)
  # ElasticNet? (mix of L1 and L2 r12n)
  # Lars (least angle regression - numerically efficient when nfeats >> ninstances)
  # LassoLars (lars + L1 r12n)
  # LogisticRegression: umbrella with a bunch of solvers. liblinear is the only
  #     one that supports l1 r12n. It uses coordinate descent.
  # SGDClassifier (this + log loss recommended for 'large datasets')
  if 0:
    clf = sklearn.svm.LinearSVC(
        C=1.0,
        penalty='l1',
        dual=False,
        verbose=2,
        max_iter=20,
        )
  elif 0:
    # Uses about 8g memory with thresh=10
    # and slow af
    clf = sklearn.linear_model.ElasticNet(
        alpha=1.0,
        l1_ratio=.7,
        max_iter=10,
    )
  elif 0:
    clf = sklearn.linear_model.Lasso(
        alpha=1.0,
        normalize=False, # ?
        max_iter=1000,
        positive=False,
        )
  elif 0:
    clf = sklearn.linear_model.LassoCV(
        n_jobs=2,
        )
  elif 0:
    clf = sklearn.Lars()
  elif 0:
    clf = sklearn.linear_model.LassoLars(
        alpha=1.0,
        verbose=1,
        max_iter=500,
        )
  elif 1:
    clf = sklearn.linear_model.SGDClassifier(
        loss='log',
        #penalty='l1',
        penalty='elasticnet',
        l1_ratio=0.5,
        alpha=1e-6, # default = 1e-4
        n_iter=82, # epochs
        shuffle=True, # shuffling between epochs
        verbose=1,
        learning_rate='optimal',
        )
  print "Built classifier {}".format(clf)

  clf.fit(X, y)

  save_model(clf)

  if args.validation_fold:
    Xval, yval = vectorize.load_fold(args.validation_fold)
    score = clf.score(Xval, yval)
    train_score = clf.score(X, y)
    print "Score on validation data = {:.2%} (train acc = {:.2%})".format(score, train_score)

  return clf

if __name__ == '__main__':
  with time_me(mode='print'):
    clf = main()
