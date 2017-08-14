#!/usr/bin/env python

"""
Generates a dataframe, with a row for each classification instance, having...
- metadata: uid, pid, label, fold
- meta-features: characteristics of the instance that might account for different 
    success rates for different predictors. Used for 'FWLS'.

This df is used by train.py, along with the logits of some number of specified models.
"""

import argparse
import pandas as pd
import numpy as np
import sklearn.preprocessing

from baskets.user_wrapper import iterate_wrapped_users
from baskets.time_me import time_me

METAVECTORS_PICKLEPATH = 'metavectors.pickle'

def rows_for_user(user, fold):
  rows = []
  order = user.orders[-1]
  orderpids = set(order.products)
  for pid in user.sorted_pids:
    label = pid in orderpids
    row = [user.uid, pid, label, fold, user.norders, user.nprods]
    rows.append(row)
  return rows


# For now, just one metafeature: norders. More to consider later:
# - nprods
# - Some measure of 'consistency' from order-to-order?
# - Commonness of pid (i.e. n occurrences in the dataset/training set)
# - number of 30 day intervals in user history
metafeature_columns = ['norders', 'nprods']
def vectorize(userfolds):
  cols = ['uid', 'pid', 'label', 'fold'] + metafeature_columns
  rows = []
  for fold, users in userfolds.iteritems():
    for user in users:
      rows += rows_for_user(user, fold)
  df = pd.DataFrame(rows, columns=cols)
  # TODO: could make 'fold' categorical to save a fair bit of space
  df['fold'] = df['fold'].astype('category')
  # Scale
  for col in metafeature_columns:
    df[col] = sklearn.preprocessing.scale(
        np.log(df[col])
        )
  return df

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('folds', nargs='+')
  args = parser.parse_args()

  users = {fold: iterate_wrapped_users(fold) for fold in args.folds}
  df = vectorize(users)
  df.to_pickle(METAVECTORS_PICKLEPATH)
  print 'Wrote df with {} metavectors'.format(len(df))

if __name__ == '__main__':
  with time_me(mode='print'):
    main()
