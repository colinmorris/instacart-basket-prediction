import os
import pandas as pd
import numpy as np
import tensorflow as tf

from baskets import common

def load_checkpoint_for_tag(tag, sess):
  cpkt_path = '{}/{}'.format(common.CHECKPOINT_DIR, tag)
  load_checkpoint(sess, cpkt_path)

def load_checkpoint(sess, checkpoint_path):
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
  saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, tag, global_step):
  model_save_path = os.path.join(common.CHECKPOINT_DIR, tag)
  if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
  saver = tf.train.Saver(tf.global_variables())
  checkpoint_path = os.path.join(model_save_path, 'vector')
  tf.logging.info('saving model %s.', checkpoint_path)
  tf.logging.info('global_step %i.', global_step)
  saver.save(sess, checkpoint_path, global_step=global_step)

_product_df = None
def load_product_df():
  global _product_df
  if _product_df is None:
    path = common.csv_path('products.csv')
    _product_df =  pd.read_csv(path,
        dtype={'product_id': np.int32, 'aisle_id': np.int32, 'department_id': np.int8},
        usecols=['product_id', 'aisle_id', 'department_id', 'product_name'],
        )
  return _product_df
