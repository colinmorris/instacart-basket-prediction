#!/usr/bin/env python
import argparse
import logging
import scipy.sparse
import numpy as np

from baskets import common, constants
from baskets.user_wrapper import iterate_wrapped_users
from baskets.time_me import time_me
import baskets.time_me

import count_pairs
from user_helpers import pids_preceding


def vectorize(users, pair_lookup):
  indptrs = [0]
  indices = []
  labels = []
  for user in users:
    order = user.orders[-1]
    orderpids = set(order.products)
    for pid in user.sorted_pids:
      label = pid in orderpids
      labels.append(label)
      # Oops, this is misnamed...
      prevs = pids_preceding(user, pid)

      # TODO: May want to normalize prevs to sum to 1? 
      # Or maybe not actually.
      prev_indices = [ (constants.N_PRODUCTS + pair_lookup[(pid, otherpid)])
          for otherpid in prevs
          if (pid, otherpid) in pair_lookup ]
      focal_ix = pid-1
      row_indices = [focal_ix] + prev_indices

      indices += row_indices
      indptrs.append( len(indices) )

  nrows = len(indptrs) - 1
  nprods = constants.N_PRODUCTS
  nfeats = nprods + len(pair_lookup)
  X_shape = (nrows, nfeats)
  dat = np.ones( len(indices) )
  X = scipy.sparse.csr_matrix( (dat, indices, indptrs), shape=X_shape )
  return X, np.array(labels, dtype=int)


def main():
  baskets.time_me.set_default_mode('print')
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('fold')
  args = parser.parse_args()

  with time_me('Loaded pair lookup'):
    lookup = count_pairs.load_pair_lookup(fold='mini') # XXX
  users = iterate_wrapped_users(args.fold)
  with time_me('Vectorized'):
    X, y = vectorize(users, lookup)
  logging.info('Loaded X with shape {} and y with shape {}'.format(
    X.shape, y.shape))
  logging.info('Mean # of non-zero features per instance = {:.1f}'.format(
    X.sum(axis=1).mean()
    ))
  # TODO: paths should be parameterized
  scipy.sparse.save_npz('vectors.npz', X)
  np.save('labels.npy', y)

  return X, y

if __name__ == '__main__':
  with time_me(mode='print'):
    X, y = main()
