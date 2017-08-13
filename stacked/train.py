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
import baskets.time_me

from metavectorize import METAVECTORS_PICKLEPATH
import metavectorize

def munge_scoreses(scoreses, df):
  npredictors = len(scoreses)
  score_shape = (len(df), npredictors)
  scores = np.empty(score_shape, dtype=np.float32)
  # Yay, nested loops :/
  i = 0
  for (uid, pid) in df[ ['uid', 'pid'] ].itertuples(index=False):
    for predictor_ix, pdict in enumerate(scoreses):
      prob = pdict[uid][pid]
      scores[i, predictor_ix] = logit(prob)
    i += 1
    
  return scores

def vectorize_fold(fold, tags, meta_df, use_metafeats=True):
  with time_me('Loaded pdicts'):
    scoreses = [common.pdict_for_tag(tag, fold) for tag in tags]
  df = meta_df[meta_df['fold']==fold]
  assert len(df)
  y = df['label']
  n_predictors = len(scoreses)
  with time_me('Munged scores for {} predictors'.format(n_predictors), mode='print'):
    # TODO: could use the logit loading fn added to user_wrapper module
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
  baskets.time_me.set_default_mode('print')
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('tags', nargs='+')
  parser.add_argument('-f', '--train-fold', default='train')
  parser.add_argument('--validation-fold', help='Fold for validation (default: None)')
  parser.add_argument('--no-metafeats', action='store_true')
  args = parser.parse_args()

  with time_me("Loaded metavectors"):
    meta_df = pd.read_pickle(METAVECTORS_PICKLEPATH)

  with time_me("Made training vectors"):
    X, y = vectorize_fold(args.train_fold, args.tags, meta_df, use_metafeats=not args.no_metafeats)
  
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
