#!/usr/bin/env python
from __future__ import division
import argparse
import os
import random
import numpy as np
import tensorflow as tf


from baskets.user_wrapper import iterate_wrapped_users
from baskets.insta_pb2 import User
from baskets import common, data_fields
from baskets.time_me import time_me

# id fields are as defined by Kaggle (i.e. starting from 1, not 0)
context_fields = [
    'pid', 'aisleid', 'deptid', 'uid', 'weight',
]
# 0 if not in previous order, otherwise order in which it was added to the 
# cart (starting from 1)
raw_feats = ['previously_ordered',]
generic_raw_feats = ['days_since_prior', 'dow', 'hour',
      'n_prev_products', 
      'n_prev_repeats', 'n_prev_reorders'
      ]
sequence_fields = ['lossmask', 'labels', ] # + raw_feats

def _seq_data(user, pids):
  """Return a tuple of (generic, product-specific) dicts
  The former's values have shape (seqlen), the latter has shape (npids, seqlen)
  """
  # TODO: maybe this could be sped up by reusing arrays on each call?
  gfs = {featname: np.empty(user.seqlen) for featname in generic_raw_feats}
  nprods = len(pids)
  pid_to_ix = dict(zip(pids, range(nprods)))
  pidfeat_shape = (nprods, user.seqlen)
  labels = np.zeros(pidfeat_shape)
  prev_ordered = np.zeros(pidfeat_shape)
  pids_seen = set([]) # unique pids seen up to but not including the ith order
  prev_pidset = None
  for i, order in enumerate(user.user.orders):
    # The order with index i corresponds to the i-1th element of the sequence
    # (we always skip the first order, because by definition it can have no
    # reorders)
    seqidx = i-1
    ordered = order.products
    unordered = set(ordered) # I made a funny
    # Calculate the generic (non-product-specific) features
    # Sometimes the value of the next order's feature is a function of this order
    if i < user.seqlen:
      gfs['n_prev_products'][i] = len(ordered)
      gfs['n_prev_reorders'][i] = len(pids_seen.intersection(unordered)) 
      gfs['n_prev_repeats'][i] = 0 if prev_pidset is None else len(prev_pidset.intersection(unordered))
    # And some features are calculated wrt the current order
    if i > 0:
      gfs['days_since_prior'][seqidx] = order.days_since_prior
      gfs['dow'][seqidx] = order.dow
      gfs['hour'][seqidx] = order.hour
    # Product specific feats
    for (cart_index, pid) in enumerate(ordered):
      try:
        j = pid_to_ix[pid]
      except KeyError:
        continue
      if i < user.seqlen:
        prev_ordered[j, i] = cart_index+1
      if i > 0:
        labels[j, seqidx] = 1
    pids_seen.update(unordered)
    prev_pidset = unordered

  lossmask = np.zeros(pidfeat_shape)
  first_occurences = prev_ordered.argmax(axis=1)
  # There's probably a clever non-loopy way to do this.
  for j, first in enumerate(first_occurences):
    lossmask[j, first:] = 1
  pidfeats = dict(labels=labels, lossmask=lossmask, previously_ordered=prev_ordered)
  return gfs, pidfeats

def get_user_sequence_examples(user, product_lookup, testmode, max_prods):
  assert not testmode
  max_prods = max_prods or float('inf')
  nprods = min(max_prods, user.nprods)
  weight = 1 / nprods
  # Generic context features
  base_context = {
      'uid': intfeat(user.uid),
      'weight': floatfeat(weight),
      }
  pids = random.sample(user.all_pids, nprods)
  genericfeats, prodfeats = _seq_data(user, pids)
  # Generic sequence features
  def to_seq_feat(tup):
    name, featarray = tup
    seqfn = floatseqfeat if data_fields.FIELD_LOOKUP[name].dtype == float else intseqfeat
    return name, seqfn(featarray)
  base_seqdict = dict(map(to_seq_feat, genericfeats.items()))
  for pidx, pid in enumerate(pids):
    # Context features (those that don't scale with seqlen)
    ctx_dict = base_context.copy()
    aisleid, deptid = product_lookup[pid-1]
    product_ctx = dict(pid=intfeat(pid), aisleid=intfeat(aisleid), deptid=intfeat(deptid))
    ctx_dict.update(product_ctx)
    context = tf.train.Features(feature=ctx_dict)
    # Sequence features
    seqdict = base_seqdict.copy()
    product_seqdict = dict(to_seq_feat( (name, featarray[pidx]) )
        for name, featarray in prodfeats.iteritems())
    seqdict.update(product_seqdict)
    feature_lists = tf.train.FeatureLists(feature_list = seqdict)
    example = tf.train.SequenceExample(
        context = context,
        feature_lists = feature_lists,
        )
    yield example

def intfeat(val):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(val)]))

def floatfeat(val):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[float(val)]))

def intlistfeat(vals):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=vals))

def floatlistfeat(vals):
  return tf.train.Feature(float_list=tf.train.FloatList(value=vals))

def intseqfeat(ints):
  return _seqfeat(ints, intfeat)

def floatseqfeat(floats):
  return _seqfeat(floats, floatfeat)

def _seqfeat(values, featfn):
  feats = [featfn(v) for v in values]
  return tf.train.FeatureList(feature=feats)

def write_user_vectors(user, writer, product_df, testmode, max_prods):
  n = 0
  for example in get_user_sequence_examples(user, product_df, testmode, max_prods):
    writer.write(example.SerializeToString())
    n += 1
  return n

def load_product_lookup():
  lookuppath = os.path.join(common.DATA_DIR, 'product_lookup.npy')
  return np.load(lookuppath)

def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('user_records_file')
  parser.add_argument('--out', help='Name to use for saved tfrecords file.\
      Defaults to a name based on input tfrecords file.')
  parser.add_argument('--test-mode', action='store_true', 
      help='Include final "testorder" in sequences, and only vectorize test users.')
  parser.add_argument('-n', '--n-users', type=int, 
      help='limit on number of users vectorized (default: none)')
  parser.add_argument('--max-prods', type=int, default=None,
      help='Max number of products to take per user (default: no limit)')
  args = parser.parse_args()
  random.seed(1337)

  if args.test_mode:
    raise NotImplemented("Sorry, come back later.")

  outpath = common.resolve_vector_recordpath(args.out or args.user_records_file)
  tf.logging.info("Writing vectors to {}".format(outpath))
  writer_options = tf.python_io.TFRecordOptions(
      compression_type=common.VECTOR_COMPRESSION_TYPE)
  writer = tf.python_io.TFRecordWriter(outpath, options=writer_options)
  prod_lookup = load_product_lookup()
  i = 0
  nseqs = 0
  for user in iterate_wrapped_users(args.user_records_file):
    nseqs += write_user_vectors(user, writer, prod_lookup, args.test_mode, args.max_prods)
    i += 1
    if args.n_users and i >= args.n_users:
      break
    if (i % 10000) == 0:
      print "i={}... ".format(i)
  print "Wrote {} sequences for {} users".format(nseqs, i)

if __name__ == '__main__':
  with time_me():
    main()
