#!/usr/bin/env python
import argparse
import tensorflow as tf
import json

import baskets
from baskets import hypers, common
from baskets.time_me import time_me
from baskets.dataset import BasketDataset

def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--recordfile', default='train.tfrecords')
  parser.add_argument('-n', type=int, default=50, help='How many batches to calculate over')
  args = parser.parse_args()

  hps = hypers.get_default_hparams()
  hps.features = baskets.feature_spec.FeatureSpec.all_features_spec().names
  hps.batch_size = 1 # So we don't have to deal with variable sequence lengths
  
  dat = BasketDataset(hps, args.recordfile)
  feats = dat['features']

  batch_count, batch_sum, batch_sos, _ = tf.nn.sufficient_statistics(feats, axes=[0, 1])
  nfeats = batch_sum.shape[0]
  acc_shape = [nfeats]
  count_acc = tf.Variable(0.0)
  sum_acc = tf.Variable(tf.zeros(acc_shape, dtype=tf.float32))
  sos_acc = tf.Variable(tf.zeros(acc_shape, dtype=tf.float32))
  update_ops = [
      tf.assign_add(count_acc, batch_count),
      tf.assign_add(sum_acc, batch_sum),
      tf.assign_add(sos_acc, batch_sos),
  ]
  mean, var = tf.nn.normalize_moments(count_acc, sum_acc, sos_acc, shift=None)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  for i in range(args.n):
    sess.run(update_ops)

  final_mean, final_var = sess.run([mean, var])

  def save_stats():
    featspec = dat.feature_spec
    featdat = {}
    offset = 0
    for feat in featspec.features:
      stats = []
      for withinfeat_index, i in enumerate(range(offset, offset+feat.arity)):
        innerstats = {}
        innerstats['mean'] = float(final_mean[i])
        innerstats['variance'] = float(final_var[i])
        stats.append(innerstats)
      featdat[feat.name] = stats
      offset += feat.arity

    with open('{}/feature_stats.json'.format(common.DATA_DIR), 'w') as f:
      json.dump(featdat, f, indent=2, separators=(',', ': '))
    
    return featdat

  save_stats()

def foo():
  x = tf.Variable([
    [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
    [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    ])
  ss = tf.nn.sufficient_statistics(x, axes=[0, 1])
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  fooz = sess.run(ss[:3])
  print fooz

if __name__ == '__main__':
  with time_me():
    main()

