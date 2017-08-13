#!/usr/bin/env python
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from scipy.special import expit

from baskets import common, constants, utils
from baskets.time_me import time_me

import count_pairs
import train

BOOST = 1

def poke_uniprods(coefs, df, n=10):
  idxs = np.argsort(coefs)
  for sign in (-1, 1):
    print
    for rank in range(n):
      meta_i = rank if sign == 1 else -1*(rank+1)
      pidx = idxs[meta_i]
      pid = pidx+1
      weight = coefs[pidx]
      if weight == 0:
        print "Hit zero weight! Stopping early"
        break
      name = df.loc[pidx, 'product_name']
      print 'Weight on {} = {:.3f}'.format(name, weight)

def poke_biprods(coefs, pair_lookup, df, n=10):
  idxs = np.argsort(coefs)
  for sign in (-1, 1):
    print
    for rank in range(n):
      meta_i = rank if sign == 1 else -1*(rank+1)
      fidx = idxs[meta_i]
      focal, other = pair_lookup[fidx]
      weight = coefs[fidx]
      focal_name, other_name = df.loc[(focal-1, other-1), 'product_name']
      print 'Weight on {} when followed by {} = {:.3f}'.format(focal_name, other_name, weight)


path = 'model.pkl'

clf = train.load_model()
print "Loaded {}".format(clf)

intercept = clf.intercept_[0]
print "Learned intercept = {:.3f} (i.e. p = {:.3f}".format(
    intercept, expit(intercept))
coef = clf.coef_

print "Loaded coefs with shape {}. {} non-zero entries".format(
    coef.shape, (coef != 0).sum())

# Unwrap outer layer
coef = coef[0]

nprods = constants.N_PRODUCTS
if BOOST:
  boost_coef = coef[0]
  print "Coef on boosted score: {:.3f}".format(boost_coef)
  offset = 1
else:
  offset = 0
uniprod_coefs = coef[offset:nprods+offset]
biprod_coefs = coef[nprods+offset:]

print "{:,} uni prod feats, {:,} bi prod feats".format(nprods, len(biprod_coefs))

for coefs, name in [ (uniprod_coefs, 'uniprod'), (biprod_coefs, 'biprod') ]:
  nz = coefs[coefs != 0]
  print "{} {} features are non-zero".format(len(nz), name)
  print "Dist of nonzero {} features...".format(name)
  s = pd.Series(nz)
  print s.describe()

with time_me("Loaded feature lookup", mode='print'):
  featmap = count_pairs.load_pair_lookup()

pair_lookup = {v: k for (k, v) in featmap.iteritems()}

prod_df = utils.load_product_df()

poke_uniprods(uniprod_coefs, prod_df)

poke_biprods(biprod_coefs, pair_lookup, prod_df, 10)
