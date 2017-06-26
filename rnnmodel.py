"""This code roughly based on magenta.models.sketch-rnn.model"""
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from tensorflow.contrib.training import HParams

from batch_helpers import UserWrapper

# TODO: stuff to add later
# - dropout
def get_default_hparams():
  return HParams(
      is_training=True,
      rnn_size=128,
      batch_size=100,
      max_seq_len=100, # TODO: not sure about this
      nfeats=UserWrapper.NFEATS,
      learning_rate=0.001, # ???
      save_every=5000,
      # Can scale this up to at least like 195k even using our dumb sampling
      # technique, since that's the # of users in the dataset
      num_steps=17000,
  )

def get_toy_hparams():
    base = get_default_hparams()
    base.num_steps=100
    return base


class RNNModel(object):

  def __init__(self, hyperparams, reuse=False):
    self.hps = hyperparams
    with tf.variable_scope('instarnn', reuse=reuse):
      self.build_model()

  def build_model(self):
    hps = self.hps
    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    # TODO: later look at LSTMCell, which enables peephole + projection layer
    # Also, this looks superficially interesting: 
    #   https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/TimeFreqLSTMCell
    self.cell = tf.contrib.rnn.BasicLSTMCell(
        hps.rnn_size,
        forget_bias=1.0,
        # TODO: maybe docs should be more clear on type of activation kwarg,
        # since this doesn't work
        #activation='tanh',
    )
    self.sequence_lengths = tf.placeholder(
        dtype=tf.int32, shape=[self.hps.batch_size], name="seqlengths",
    )
    self.input_data = tf.placeholder(
        dtype=tf.float32,
        shape=[hps.batch_size, hps.max_seq_len, hps.nfeats],
        name="input",
    )
    label_shape = [hps.batch_size, hps.max_seq_len]
    self.labels = tf.placeholder(
            # TODO: idk about this dtype stuff
            dtype=tf.float32, shape=label_shape, name="labels",
    )

    self.initial_state = self.cell.zero_state(batch_size=hps.batch_size,
            dtype=tf.float32)
    
    output, last_state = tf.nn.dynamic_rnn(
            self.cell, 
            self.input_data,
            sequence_length=self.sequence_lengths,
            # this kwarg is optional, but docs aren't really clear on what
            # happens if it isn't provided. Probably just the zero state,
            # so this isn't necessary. But whatever.
            # yeah, source says zeros. But TODO should prooobably be documented?
            initial_state=self.initial_state,
            dtype=tf.float32,
    )

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.hps.rnn_size, 1])
      output_b = tf.get_variable('output_b', [1])
    
    output = tf.reshape(output, [-1, hps.rnn_size])
    logits = tf.nn.xw_plus_b(output, output_w, output_b)
    logits = tf.reshape(logits, label_shape)
    self.logits = logits
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits)
    # TODO: does the fact that the mean here includes a variable number of dummy
    # values (from padding to max seq len) change anything? Need to be a bit careful.
    self.cost = tf.reduce_mean(loss)

    if self.hps.is_training:
        self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(
                self.cost,
                self.global_step,
        )
