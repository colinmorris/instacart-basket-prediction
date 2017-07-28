#!/usr/bin/env python

from __future__ import division
import argparse
from collections import defaultdict
import os
import random
import numpy as np
from numpy.lib import recfunctions
import logging
import math

from baskets.user_wrapper import iterate_wrapped_users
from baskets.insta_pb2 import User
from baskets import common, data_fields
from baskets.time_me import time_me

import fields

DEBUG = False

# (These are just sort of made up)
# TODO: 30 days really means "30+ days" and so should be taken with a grain
# of salt. Could even consider translating 30 day intervals into some bigger
# number. Coming up with a principled choice is kinda tricky. Could look at 
# histogram and try to extrapolate the mean of the tail that got gobbled up.
_FRECENCY_HALFLIFE_DAYS = 7
FRECENCY_DAYS_LAMBDA = math.log(2) / _FRECENCY_HALFLIFE_DAYS
_FRECENCY_HALFLIFE_ORDERS = 4
FRECENCY_ORDERS_LAMBDA = math.log(2) / _FRECENCY_HALFLIFE_ORDERS


def _write_all_recarray_fields(recarr, value):
  view = recarr.view(fields.dtype).reshape(len(recarr), -1)
  view[:] = value

_generic_dtypes = [(featname, fields.dtype) for featname in fields.generic_raw_feats]
_product_dtypes = [(featname, fields.dtype) for featname in fields.product_raw_feats]
def _order_data(user, pids, product_lookup, order_idx=-1, test=False):
  """Return a tuple of (generic, prod-specific) recarrays.
  """
  assert not test
  assert order_idx != 0
  pid_to_ix = {pid: i for (i, pid) in enumerate(pids)}
  # XXX: Using the setattr syntax on recarrays is a little dangerous, because 
  # you'll get no indication of the attribute you're setting isn't a named field.
  # Also, it's a mild perf annotance. Somewhere around 1/4 to 1/3 of total run time of vectorization is spent in setattr.
  g = np.recarray( (1,), dtype=_generic_dtypes )
  p = np.recarray( (len(pids),), dtype=_product_dtypes )
  # Set a special 'not initialized' value for bookkeeping
  NOTSET = -1337
  _write_all_recarray_fields(g, NOTSET)
  _write_all_recarray_fields(p, NOTSET)

  g['uid'] = user.uid
  g['user_prods'] = len(pids)
  g['n_prev_orders'] = len(user.user.orders[:order_idx])

  # focal order
  order = user.user.orders[order_idx]
  ops = order.products
  opset = set(ops)
  prev = user.user.orders[order_idx-1]
  prev_ops = prev.products
  prev_opset = set(prev_ops)
  for passthrough_feat in ['orderid', 'dow', 'hour', 'days_since_prior']:
    g[passthrough_feat] = getattr(order, passthrough_feat)
  g['prev_order_size'] = len(prev.products)
  
  # prevprev feats
  try:
    prevprev = user.user.orders[order_idx-2]
  except IndexError:
    prevprev = None
  if prevprev is None:
    for prevprev_feat in ['n_prev_repeats']:
      g[prevprev_feat] = np.nan
  else:
    prevprev_opset = set(prevprev.products)
    repeats = prev_opset.intersection(prevprev_opset)
    g['n_prev_repeats'] = len(repeats)

  for pi, pid in enumerate(pids):
    pp = p[pi]
    # (numpy will implictly convert, so that's nice)
    pp['label'] = pid in opset
    pp['prev_cartorder'] = np.nan if pid not in prev_opset else list(prev_ops).index(pid) 
    pp['pid'] = pid
    pp['aisleid'], pp['deptid'] = product_lookup[pid-1]

  zero_init_cols = ['n_prev_focals', 'n_prev_focals_this_dow', 'n_prev_focals_this_hour',
      'frecency_days', 'frecency_orders', 
      'n_consecutive_prev_focal_orders']
  for col in zero_init_cols:
    p[col] = 0
  # running count of days/orders separating loop order from the focal order
  daycount = order.days_since_prior
  ordercount = 1
  order_sizes = []
  pid_order_sizes = {pid: [] for pid in pids}
  streakers = set(pids)
  frozen_fields = defaultdict(set)
  def freeze(pid, field, value):
    if field in frozen_fields[pid]:
      return
    p[pid_to_ix[pid]][field] = value
    frozen_fields[pid].add(field)

  # Walk backwards starting from the previous order
  for prevo in user.user.orders[order_idx-1::-1]:
    osize = len(prevo.products)
    order_frecency_days = math.exp(-1 * FRECENCY_DAYS_LAMBDA * daycount)
    order_frecency_orders = math.exp(-1 * FRECENCY_ORDERS_LAMBDA * ordercount)
    for cartorder, pid in enumerate(prevo.products):
      try:
        pi = pid_to_ix[pid]
      except KeyError:
        continue
      pp = p[pi]
      pp['n_prev_focals'] += 1
      pp['n_prev_focals_this_dow'] += prevo.dow == order.dow
      pp['n_prev_focals_this_hour'] += prevo.hour == order.hour
      pp['frecency_days'] += order_frecency_days
      pp['frecency_orders'] += order_frecency_orders
      if pid in streakers:
        pp['n_consecutive_prev_focal_orders'] += 1

      freeze(pid, 'last_focal_cartorder', cartorder)
      freeze(pid, 'orders_since_focal', ordercount)
      freeze(pid, 'days_since_focal', daycount)

      pid_order_sizes[pid].append(osize)
      # (The below feats may of course be overwritten later)
      pp['orders_since_first_focal'] = ordercount
      pp['days_since_first_focal'] = daycount

    daycount += prevo.days_since_prior
    ordercount += 1
    order_sizes.append(osize)
    prevo_prodset = set(prevo.products)
    streakers = streakers.intersection(prevo_prodset)

  g['avg_order_size'] = np.mean(order_sizes)
  for pid, osizes in pid_order_sizes.iteritems():
    p[pid_to_ix[pid]].avg_focal_order_size = np.mean(osizes)

  if DEBUG:
    for pfield in fields.product_raw_feats:
      if (p[pfield] == NOTSET).any():
        logging.warning('Field {} not set'.format(pfield))
    for gfield in fields.generic_raw_feats:
      if (g[gfield] == NOTSET).any():
        logging.warning('Field {} not set'.format(gfield))
  return g, p

def get_user_vectors(user, max_prods, product_lookup, testmode):
  assert not testmode
  max_prods = max_prods or float('inf')
  nprods = min(max_prods, user.nprods)
  pids = random.sample(user.all_pids, nprods)
  generic_feats, prod_feats = _order_data(user, pids, product_lookup, order_idx=-1, test=testmode)
  assert prod_feats.shape[0] == nprods
  generic_feats = np.tile(generic_feats, [nprods])
  # Can't use concatenate for recarrays with diff dtypes
  ret = recfunctions.merge_arrays([generic_feats, prod_feats], asrecarray=True, flatten=True)
  return ret

def accumulate_user_vectors(users, max_prods, product_lookup, max_users, testmode):
  BUFFER_SIZE = float('inf') # XXX: see what mem usage looks like before over-engineering
  vec_accumulator = []
  nusers = 0
  for user in users:
    vecs = get_user_vectors(user, max_prods, product_lookup, testmode)
    vec_accumulator.append(vecs)
    nusers += 1
    if max_users and nusers >= max_users:
      break
    if nusers % 10000 == 0:
      print "{}... ".format(nusers)

  print "Accumulated vectors for {} users".format(len(vec_accumulator))
  concatted = np.concatenate(vec_accumulator)
  final_arr = concatted.view(np.recarray)
  return final_arr

def load_product_lookup():
  lookuppath = os.path.join(common.DATA_DIR, 'product_lookup.npy')
  return np.load(lookuppath)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('user_records_file')
  parser.add_argument('-n', '--n-users', type=int, 
      help='limit on number of users vectorized (default: none)')
  parser.add_argument('--max-prods', type=int, default=None,
      help='Max number of products to take per user (default: no limit)')
  parser.add_argument('--testmode', action='store_true')
  parser.add_argument('--outname', help='Identifier used to name the generated\
      npy file. Default name is based on user_records_file')
  args = parser.parse_args()
  random.seed(1337)

  if args.testmode:
    raise NotImplemented("Sorry, come back later.")

  prod_lookup = load_product_lookup()
  user_iter = iterate_wrapped_users(args.user_records_file)
  vecs = accumulate_user_vectors(user_iter, args.max_prods, prod_lookup, 
      args.n_users, args.testmode)
  output_tag = args.outname or args.user_records_file
  outpath = common.resolve_scalarvector_path(output_tag)
  np.save(outpath, vecs)

if __name__ == '__main__':
  with time_me(mode='print'):
    foo = main()
