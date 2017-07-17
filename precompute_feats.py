from __future__ import division
import argparse
import time
import pandas as pd
import numpy as np
import tensorflow as tf


from baskets.batch_helpers import iterate_wrapped_users
from baskets.insta_pb2 import User
from baskets import common, utils

context_fields = [
    'pid', 'aisleid', 'deptid', 'uid', 'seqlen', 'weight',
]
raw_feats = ['previously_ordered', 'days_since_prior', 'dow', 'hour',
      'n_prev_products', 'n_prev_repeats', 'n_prev_reorders']
sequence_fields = ['lossmask', 'labels', ] # + raw_feats

def write_user_vectors(user, writer, product_df, testmode, max_prods):
  assert not testmode
  assert max_prods, "You should set a limit for now..."
  nprods = min(max_prods, user.nprods)
  weight = 1 / nprods
  base_context = {
      'uid': intfeat(user.uid),
      'seqlen': intfeat(user.seqlen),
      'weight': floatfeat(weight),
      }
  base_features = []
  for pidx, pid in enumerate(pids):
    # Add pid, aisleid, deptid to context
    feats = [ 
        floatlistfeat(base_features[i] + prod_feats[pidx, i])
        for i in range(seqlen)
    ]
    seqfeats = {
        'label': seq_featlist(labels[pidx], floats=1),
        'lossmask': seq_featlist(lossmask[pidx], floats=1),
        'feats': tf.train.FeatureList(feature = feats ) 
    }
    feature_lists = tf.train.FeatureLists(feature_list = seqfeats)
    example = tf.train.SequenceExample(
        context = context,
        feature_lists = feature_lists,
        )



def intfeat(val):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))

def floatfeat(val):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[val]))

def intlistfeat(vals):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=vals))

def floatlistfeat(vals):
  return tf.train.Feature(float_list=tf.train.FloatList(value=vals))

def write_user_vectors(user, writer, product_df, testmode, max_prods):
  istest = int(user.user.test)
  seqlen = user.seqlen
  weight = 1 / len(user.all_pids)
  for pid in user.all_pids:
    aisleid, deptid = product_df.loc[pid, ['aisle_id', 'department_id']]
    ctx_features = {
        'pid': scalarfeat(pid),
        'aisleid': scalarfeat(aisleid),
        'deptid': scalarfeat(deptid),
        # TODO: not sure about this one. Not useful during training. Needed
        # during eval (i.e. when generating pdicts)
        # So maybe allow different modes via command line options that 
        # control certain features?
        # TODO: OTOH, I guess it doesn't really hurt that much to have
        # a couple extra fields, and it might make for some less finnicky code.
        'uid': scalarfeat(user.uid),
        'seqlen': scalarfeat( seqlen ),
        'weight': scalarfloatfeat( weight ),
    }
    # Feature: an array of values
    # Features: map from name to Feature
    context = tf.train.Features(feature=ctx_features)
    # FeatureList: List of Feature (presumably per frame/timestep)
    # FeatureLists: Map from name to FeatureList
    ts = user.training_sequence_for_pid(pid, maxlen=-1, testmode=mode=='test')
    labels = ts['labels']
    lossmask = ts['lossmask']
    # TODO: should save names of feats used and their order somewhere.
    # switching between diff featuresets and backwards compat could be a nightmare
    # Could use a unique name in the FeatureLists map per feature, but that seems unwieldy
    # Maybe do some kind of janky versioning in features.py
    featdat = ts['x']
    feats = []
    for i in range(featdat.shape[0]):
      feats.append( floatlistfeat(featdat[i]) )
    # TODO: could get rid of label for all modes but train/eval,
    # and lossmask for all but train
    seqfeats = {
        'label': seq_featlist(labels, floats=1),
        'lossmask': seq_featlist(lossmask, floats=1),
        'feats': tf.train.FeatureList(feature = feats ) 
    }
    feature_lists = tf.train.FeatureLists(feature_list = seqfeats)
    # A SequenceExample has a context Features
    # and a FeatureLists
    example = tf.train.SequenceExample(
        context = context,
        feature_lists = feature_lists,
        )
    # (A normal Example has just a Features)
    writer.write(example.SerializeToString())

def seq_featlist(seqdat, floats=False):
  scalarfn = scalarfloatfeat if floats else scalarfeat
  feats = [scalarfn(x) for x in seqdat]
  return tf.train.FeatureList(feature=feats)
  # (not feature=[intlistfeat(seqdat)] as I previously thought)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('user_records_file')
  parser.add_argument('--test-mode', action='store_true', 
      help='Include final "testorder" in sequences, and only vectorize test users.')
  parser.add_argument('-n', '--n-users', type=int, 
      help='limit on number of users vectorized (default: none)')
  parser.add_argument('--max-prods', type=int, 
      help='Max number of products to take per user (default: None)')
  args = parser.parse_args()

  if args.test_mode:
    raise NotImplemented("Sorry, come back later.")

  outpath = common.path_for_vectors(args.user_records_file)
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
