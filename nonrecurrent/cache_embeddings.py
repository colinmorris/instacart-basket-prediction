import argparse
import numpy as np
import tensorflow as tf

from baskets import rnnmodel, utils, hypers, dataset, common
from baskets.time_me import time_me

def path_for_cached_embeddings(tag):
  return os.path.join(common.XGBOOST_DIR, 'cache', 'embeddings_{}.npy'.format(tag))

def load_embeddings(tag):
  return np.load(path_for_cached_embeddings(tag))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('tag')
  args = parser.parse_args()
  tag = args.tag

  hps = hypers.hps_for_tag(tag)
  hps.is_training = 0
  hps.batch_size = 1
  # (dummy dataset, just so we have some placeholder values for the rnnmodel's input vars)
  dat = dataset.BasketDataset(hps, 'unit_tests.tfrecords')
  model = rnnmodel.RNNModel(hps, dat)
  sess = tf.InteractiveSession()
  utils.load_checkpoint_for_tag(sess, tag)

  def lookup(varname):
    with tf.variable_scope('instarnn', reuse=True):
        var = tf.get_variable(varname)
    val = sess.run(var)
    return val

  emb = lookup('product_embeddings')
  outpath = path_for_cached_embeddings(tag)
  np.save(outpath, emb)
  print 'Saved embeddings with shape {} to {}'.format(emb.shape, outpath)


if __name__ == '__main__':
  with time_me(mode='print'):
    main()
