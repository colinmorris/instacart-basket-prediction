"""This code roughly based on magenta.models.sketch-rnn.model"""
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from tensorflow.contrib.training import HParams

from batch_helpers import UserWrapper
from features import NFEATS
from constants import N_PRODUCTS

# TODO: stuff to add later
# - dropout
def get_default_hparams():
  return HParams(
      is_training=True,
      rnn_size=128,
      batch_size=100,
      max_seq_len=100, # TODO: not sure about this
      nfeats=NFEATS,
      learning_rate=0.001, # ???
      #decay_rate=0.9999,
      decay_rate=0.99999, # set to 1 to disable lr decay
      min_learning_rate=0.00001,
      save_every=5000,
      eval_every=1000,
      # There are about 195k users in the dataset, so if we take on sequence
      # from each, it'd take about 2k steps to cycle through them all
      num_steps=17000,
      product_embeddings=False,
      product_embedding_size=64,
      grad_clip=0.0, # gradient clipping. Set to falsy value to disable.
      embedding_l2_cost=.0001,
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
    cell_input = self.input_data
    if hps.product_embeddings:
      product_embeddings = tf.get_variable("product_embeddings",
          [N_PRODUCTS, hps.product_embedding_size])
      self.product_ids = tf.placeholder(
          dtype=tf.int32, shape=[self.hps.batch_size], name="product_ids"
      )
      lookuped = tf.nn.embedding_lookup(
          product_embeddings,
          self.product_ids,
          max_norm=None, # TODO: experiment with this param
      )
      lookuped = tf.reshape(lookuped, [self.hps.batch_size, 1, hps.product_embedding_size])
      lookuped = tf.tile(lookuped, [1, self.hps.max_seq_len, 1])
      cell_input = tf.concat([self.input_data, lookuped], 2)
    
    label_shape = [hps.batch_size, hps.max_seq_len]
    self.labels = tf.placeholder(
            # TODO: idk about this dtype stuff
            dtype=tf.float32, shape=label_shape, name="labels",
    )
    self.lossmask = tf.placeholder(
        dtype=tf.float32, shape=label_shape, name='lossmask',
    )

    self.initial_state = self.cell.zero_state(batch_size=hps.batch_size,
            dtype=tf.float32)
    
    output, last_state = tf.nn.dynamic_rnn(
            self.cell, 
            cell_input,
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
    # apply loss mask
    loss = tf.multiply(loss, self.lossmask)
    # Total loss per sequence
    loss_per_seq = tf.reduce_sum(loss, axis=1)
    # tf has way too many ways to do division :[
    loss_per_seq = tf.realdiv(loss_per_seq, 
        tf.reduce_sum(self.lossmask, axis=1)
    )
    # TODO: does the fact that the mean here includes a variable number of dummy
    # values (from padding to max seq len) change anything? Need to be a bit careful.
    # Maybe need to do it in 2 steps? inner avgs. then outer avg-of-avgs?
    # Or just weight by seqlens.
    
    # TODO XXX I think I had the right idea with this, but right now it's producing
    # nans. I guess sometimes lossmask sums to 0? Need to look more into this later.
    #self.cost = tf.reduce_mean(loss_per_seq)
    self.cost = tf.reduce_mean(loss)
    if self.hps.product_embeddings:
      embedding_weight_penalty = self.hps.embedding_l2_cost * tf.nn.l2_loss(product_embeddings)
      self.cost = tf.add(self.cost, embedding_weight_penalty)

    if self.hps.is_training:
        self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)
        if self.hps.grad_clip:
          gvs = optimizer.compute_gradients(self.cost)
          g = self.hps.grad_clip
          capped_gvs = [ (tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
          self.train_op = optimizer.apply_gradients(
              capped_gvs, global_step=self.global_step
          )
        else:
          self.train_op = optimizer.minimize(
                  self.cost,
                  self.global_step,
          )
