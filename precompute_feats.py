from __future__ import division
import argparse
import random
import time
import pandas as pd
import numpy as np
import tensorflow as tf


from baskets.batch_helpers import iterate_wrapped_users
from baskets.insta_pb2 import User
from baskets import common, utils
from baskets.time_me import time_me

context_fields = [
    'pid', 'aisleid', 'deptid', 'uid', 'weight',
]
raw_feats = ['previously_ordered',]
generic_raw_feats = ['days_since_prior', 'dow', 'hour',
      'n_prev_products',] 
# Going to skip these last two for now. Slightly annoying bookeeping to 
# calculate them, and not convinced they really help much.
#'n_prev_repeats', 'n_prev_reorders']
sequence_fields = ['lossmask', 'labels', ] # + raw_feats

# TODO: unit test me
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
  for i, order in enumerate(user.user.orders):
    # The order with index i corresponds to the i-1th element of the sequence
    # (we always skip the first order, because by definition it can have no
    # reorders)
    seqidx = i-1
    ordered = order.products
    # Calculate the generic (non-product-specific) features
    if i < user.seqlen:
      gfs['n_prev_products'][i] = len(ordered)
    if i > 0:
      gfs['days_since_prior'][seqidx] = order.days_since_prior
      gfs['dow'][seqidx] = order.dow
      gfs['hour'][seqidx] = order.hour
    # Product specific feats
    for pid in ordered:
      try:
        j = pid_to_ix[pid]
      except KeyError:
        continue
      if i < user.seqlen:
        prev_ordered[j, i] = 1
      if i > 0:
        labels[j, seqidx] = 1

  lossmask = np.zeros(pidfeat_shape)
  first_occurences = prev_ordered.argmax(axis=1)
  # There's probably a clever non-loopy way to do this.
  for j, first in enumerate(first_occurences):
    lossmask[j, first:] = 1
  pidfeats = dict(labels=labels, lossmask=lossmask, prev_ordered=prev_ordered)
  return gfs, pidfeats


def write_user_vectors(user, writer, product_df, testmode, max_prods):
  for example in get_user_sequence_examples(user, product_df, testmode, max_prods):
    writer.write(example.SerializeToString())

def get_user_sequence_examples(user, product_df, testmode, max_prods):
  assert not testmode
  assert max_prods, "You should set a limit for now..."
  nprods = min(max_prods, user.nprods)
  weight = 1 / nprods
  # Generic context features
  base_context = {
      'uid': intfeat(user.uid),
      #'seqlen': intfeat(user.seqlen),
      'weight': floatfeat(weight),
      }
  pids = random.sample(user.all_pids, nprods)
  genericfeats, prodfeats = _seq_data(user, pids)
  # Generic sequence features
  base_seqdict = {name: intseqfeat(featarray) for name, featarray in genericfeats.iteritems()} 
  for pidx, pid in enumerate(pids):
    # Context features (those that don't scale with seqlen)
    ctx_dict = base_context.copy()
    aisleid, deptid = product_df.loc[pid, ['aisle_id', 'department_id']]
    product_ctx = dict(pid=intfeat(pid), aisleid=intfeat(aisleid), deptid=intfeat(deptid))
    ctx_dict.update(product_ctx)
    context = tf.train.Features(feature=ctx_dict)
    # Sequence features
    seqdict = base_seqdict.copy()
    product_seqdict = {name: intseqfeat(featarray[pidx]) 
        for name, featarray in prodfeats.iteritems()}
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


def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('user_records_file')
  parser.add_argument('--test-mode', action='store_true', 
      help='Include final "testorder" in sequences, and only vectorize test users.')
  parser.add_argument('-n', '--n-users', type=int, 
      help='limit on number of users vectorized (default: none)')
  parser.add_argument('--max-prods', type=int, default=5,
      help='Max number of products to take per user (default: 5)')
  args = parser.parse_args()

  if args.test_mode:
    raise NotImplemented("Sorry, come back later.")

  outpath = common.resolve_vector_recordpath(args.user_records_file)
  tf.logging.info("Writing vectors to {}".format(outpath))
  writer = tf.python_io.TFRecordWriter(outpath)
  product_df = utils.load_product_df()
  i = 0
  nseqs = 0
  for user in iterate_wrapped_users(args.user_records_file):
    write_user_vectors(user, writer, product_df, args.test_mode, args.max_prods)
    i += 1
    nseqs += len(user.all_pids)
    if args.n_users and i >= args.n_users:
      break
  print "Wrote {} sequences for {} users".format(nseqs, i)

if __name__ == '__main__':
  with time_me():
    main()
