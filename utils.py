import os

import tensorflow as tf

CHECKPOINT_DIR = 'checkpoints'

def load_checkpoint(sess, checkpoint_path):
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
  saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, identifier, global_step):
  if isinstance(identifier, int):
    identifier = str(identifier)
  model_save_path = os.path.join(CHECKPOINT_DIR, identifier)
  if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
  saver = tf.train.Saver(tf.global_variables())
  checkpoint_path = os.path.join(model_save_path, 'vector')
  tf.logging.info('saving model %s.', checkpoint_path)
  tf.logging.info('global_step %i.', global_step)
  saver.save(sess, checkpoint_path, global_step=global_step)
