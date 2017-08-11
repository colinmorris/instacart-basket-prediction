#!/usr/bin/env python
import argparse
import logging
import cPickle
import scipy.sparse
import numpy as np
import itertools

from baskets import common, constants
from baskets.user_wrapper import iterate_wrapped_users
import baskets.time_me
from baskets.time_me import time_me

from user_helpers import paired_pids

def pair_lookup_path(fold):
  return 'pair_lookup_{}.pickle'.format(fold)

# I mean, threshold should probably be encoded in the paths as well. But this is okay for now.
def save_pair_lookup(lookup, fold='train'):
  path = pair_lookup_path(fold)
  with open(path, 'wb') as f:
    cPickle.dump(lookup, f, 2)

# Generally should be using the same one of these for all purposes - ability 
# to parameterize by fold is mostly for the sake of testing.
def load_pair_lookup(fold='train'):
  path = pair_lookup_path(fold)
  with open(path, 'rb') as f:
    return cPickle.load(f)

def cooc_matrix_path(fold):
  return 'cooc_{}.npz'.format(fold)

def load_cooc_matrix(fold):
  return scipy.sparse.load_npz(cooc_matrix_path(fold))

def save_cooc_matrix(fold, M):
  if isinstance(M, scipy.sparse.dok_matrix):
    with time_me('Converted M to coo'):
      M = scipy.sparse.coo_matrix(M)
  scipy.sparse.save_npz(cooc_matrix_path(fold), M)

def build_cooc_matrix(users):
  """Matrix where M[a,b] is the count of co-occurrences of pids a and b,
  (or rather, the number of times b comes after the last occurrence of a - so
  not symmetrical)
  """
  nprods = constants.N_PRODUCTS
  M = scipy.sparse.dok_matrix((nprods, nprods), dtype=np.int32)
  i = 0
  for user in users:
    order = user.orders[-1]
    for pid in user.sorted_pids:
      focal_ix = pid-1
      prevs = paired_pids(user, pid)
      for prev in prevs:
        key = (focal_ix, prev-1)
        #n = M.get(key, 0)
        # further centi-optimization
        n = dict.get(M, key, 0)
        M.update({key:n+1})
        # Above is like 5x faster than below (and this inner loop is current bottleneck)
        #M[focal_ix, prev-1] += 1
    i += 1
    if i % 10000 == 0:
      logging.info('Processed {} users'.format(i))

  return M

def get_pair_lookup(fold, threshold=1, force=False):
  """
  Return a dictionary of form
    (focal, other) -> idx
  idx is the index of the feature representing 'product other came after the last
  occurrence of product focal'. (idx is 'relative' to the set of product pair features,
  and might not correspond to the same 'global' feature index in the ultimate train matrix

  The only pairs of products that make it into the returned dicts are ones that occur
  at least threshold times.
  """
  try:
    assert not force # this is some hacky control flow right here
    M = load_cooc_matrix(fold)
    logging.info('Loaded cooc matrix')
  except (AssertionError, IOError):
    logging.info('Building cooc matrix')
    users = iterate_wrapped_users(fold)
    with time_me('Built cooc matrix', mode='print'):
      M = build_cooc_matrix(users)
      M = M.tocoo()
    # converting to coo at this point might provide some memory relief
    save_cooc_matrix(fold, M)

  logging.info('Loaded cooc matrix with {:,} nonzero entries'.format(M.nnz))
  next_ix = 0
  lookup = {}
  # Turns out using scipy.sparse.find is slow af
  #with time_me('finded'):
  #  focals, others, counts = scipy.sparse.find(M)
  assert isinstance(M, scipy.sparse.coo_matrix)
  focals, others, counts = M.row, M.col, M.data
  #for i in xrange(len(counts)):
  #  focal_pidx = focals[i]
  #  other_pidx = others[i]
  #  count = counts[i]
  for (focal_pidx, other_pidx, count) in itertools.izip(focals, others, counts):

    if count >= threshold:
      k = (focal_pidx+1, other_pidx+1)
      lookup[k] = next_ix
      next_ix += 1
      if next_ix % 10**6 == 0:
        print '{:,}... '.format(next_ix)

  logging.info('Went from {:,} non-zero entries to {:,} pairs after thresholding by {}'.format(
    M.nnz, next_ix, threshold))
  if next_ix + constants.N_PRODUCTS > 2**31:
    logging.warn("Too many features even after thresholding. You're gonna have a bad time.")
  return lookup

def main():
  baskets.time_me.set_default_mode('print')
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--fold', default='train', help='Default: train')
  parser.add_argument('-t', '--threshold', type=int, default=1)
  args = parser.parse_args()

  with time_me('Made pair lookup', mode='print'):
    lookup = get_pair_lookup(args.fold, threshold=args.threshold)

  save_pair_lookup(lookup, args.fold)

if __name__ == '__main__':
  with time_me(mode='print'):
    main()
