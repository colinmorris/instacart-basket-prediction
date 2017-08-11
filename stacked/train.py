#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from scipy.special import logit

from baskets import common
from baskets.time_me import time_me

from metavectorize import METAVECTORS_PICKLEPATH
import metavectorize

def munge_scoreses(scoreses, df):
  npredictors = len(scoreses)
  score_shape = (len(df), npredictors)
  scores = np.empty(score_shape, dtype=np.float32)
  # Yay, nested loops :/
  for (i, uid, pid) in df[ ['uid', 'pid'] ].itertuples(index=True):
    for predictor_ix, pdict in enumerate(scoreses):
      prob = pdict[uid][pid]
      scores[i, predictor_ix] = logit(prob)
    
  return scores

def vectorize_fold(fold, tags, meta_df, use_metafeats=True):
  scoreses = [common.pdict_for_tag(tag, args.train_fold) for tag in tags]
  df = meta_df[meta_df['fold']==fold]
  assert len(df)
  y = df['label']
  n_predictors = len(scoreses)
  with time_me('Loaded scores for {} predictors'.format(n_predictors), mode='print'):
    scores = munge_scoreses(scoreses, df)
  if not use_metafeats:
    X = scores
  else:
    meta_cols = metavectorize.metafeature_columns
    meta = df[meta_cols].values
    # Special f_0 dummy meta feature for learning vanilla weight term per predictor
    metafeats = np.hstack([np.ones( (len(df), 1) ), meta])
    # Oh fuck this, I've spent too long trying to understand np.einsum...
    # (Worth noting that sklearn.preprocessing has a 'PolynomialFeatures' utility
    # that might have been useful here. But this is fine.)
    n_metafeats = metafeats.shape[1]
    logging.info('{} predictors x {} metafeatures -> {} coefs'.format(
      n_predictors, n_metafeats, n_predictors*n_metafeats))
    # X is 'metafeat major'. i.e. the first n_p values for each vector are the 
    # raw scores for each predictor, they're followed by each predictor's score
    # multiplied by the first metafeature and so on.
    X = np.tile(scores, n_metafeats) * np.repeat(metafeats, n_predictors, axis=1)
  return X, y


def main():
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('tags', nargs='+')
  parser.add_argument('-f', '--train-fold', default='train')
  parser.add_argument('--validation-fold')
  parser.add_argument('--no-metafeats', action='store_true')
  args = parser.parse_args()

  # load pdicts
  scores = [common.pdict_for_tag(tag, args.train_fold) for tag in args.tags]

  meta_df = pd.read_pickle(METAVECTORS_PICKLEPATH)

  X, y = vectorize_fold(args.train_fold, scores, meta_df, use_metafeats=not args.no_metafeats)
  
  # TODO: C, max_iter
  model = LogisticRegression(verbose=1)
  with time_me('Trained model', mode='print'):
    model.fit(X, y)

  model_fname = 'model.pkl'
  joblib.dump(model, model_fname)
  return model
  # TODO: report acc on validation set

if __name__ == '__main__':
  with time_me(mode='print'):
    model = main()
    print 'Learned intercept: {}\ncoefficients: {}'.format(
        model.intercept_, model.coef_
        )
