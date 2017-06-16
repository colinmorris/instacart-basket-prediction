import numpy as np
import tensorflow as tf

from tensorflow.contrib.training import HParams

# TODO: stuff to add later
# - dropout
def get_default_hparams():
  return HParams(
    is_training=True,
    rnn_size=128,
    batch_size=100,
  )


class RNNModel(object):

  def __init__(self, hyperparams, reuse=False):
  self.hps = hyperparams
  with tf.variable_scope('instarnn', reuse=reuse):
    self.build_model()

  def build_model(self):
  hps = self.hps
  if hps.is_training:
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

  self.cell = tf.contrib.rnn.BasicLSTMCell(
    hps.rnn_size,
    forget_bias=1.0,
    activation='tanh',
  )

