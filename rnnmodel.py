"""This code roughly based on magenta.models.sketch-rnn.model"""
from __future__ import absolute_import, division

import numpy as np
import tensorflow as tf

from tensorflow.contrib.training import HParams

import rnn
from batch_helpers import UserWrapper
from features import NFEATS, FEATURES
from constants import N_PRODUCTS, N_AISLES, N_DEPARTMENTS

"""TODO: there are really two distinct kinds of parameters conflated here.
1) Model parameters, which are immutable, e.g. rnn_size, feats, product_embedding_size
2) Run parameters. We can load a model from a checkpoint, and train it some more with
   different values from what it was previously trained with. e.g. learning_rate, decay_rate,
   log_every, num_steps.
Probably makes sense to store them separately.
"""
def get_default_hparams():
  return HParams(
      is_training=True,
      # 256 seems like the sweet spot for this. 512 is veeeery slow, and seems to have bad convergence
      rnn_size=256,
      batch_size=100,
      max_seq_len=100, # TODO: not sure about this
      # More correctly, the dimensionality of the feature space (there are 
      # some "features" that correspond to 2+ numbers, e.g. onehot day of week
      nfeats=NFEATS,
      feats=[f.name for f in FEATURES],
      learning_rate=0.001, # ???
      #decay_rate=0.9999,
      decay_rate=0.9999, # set to 1 to disable lr decay
      min_learning_rate=0.00001,
      save_every=5000,
      eval_every=2000,
      log_every=500,
      # There are about 195k users in the dataset, so if we take on sequence
      # from each, it'd take about 2k steps to cycle through them all
      num_steps=10000,
      product_embeddings=True, # XXX: deprecated. Set size to 0 instead.
      product_embedding_size=32,
      # Embeddings for aisle and department (22 depts, 135 aisles in dataset)
      # Set to 0 to disable these embeddings.
      aisle_embedding_size=8,
      dept_embedding_size=4,
      grad_clip=0.0, # gradient clipping. Set to falsy value to disable.
      # Did a run with weight = .0001 and that seemed too strong.
      # Mean l1 weight of embeddings was .01, max=.4. Mean l2 norm = .005 
      embedding_l2_cost=.00001, # XXX: deprecated
      l2_weight=.00001,
      # TODO: not sure if above is actually doing much now that I think
      # about it? Given some overfitted model, what's stopping it from just
      # dividing all the embeddings by 10, and multiplying all the weights
      # from the embedding to the rnn by 10? Seems I should penalize those too?
      use_recurrent_dropout=True,
      # XXX: this is the *keep* prob
      recurrent_dropout_prob=.9,
      cell='lstm', # One of lstm, layer_norm, or hyper (all from rnn.py) or basiclstm

      fully_specified=False, # Used for config file bookkeeping
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
    if hps.cell == 'lstm':
      cellfn = rnn.LSTMCell
    elif hps.cell == 'basiclstm':
      cellfn = tf.nn.rnn_cell.BasicLSTMCell
      if hps.use_recurrent_dropout:
        tf.logging.warning('Dropout not implemented for cell type basiclstm')
    elif hps.cell == 'layer_norm':
      cellfn = rnn.LayerNormLSTMCell
    elif hps.cell == 'hyper':
      cellfn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a *respectable* cell type'
    cell_kwargs = dict(forget_bias=1.0)
    if hps.cell != 'basiclstm':
      cell_kwargs['use_recurrent_dropout'] = hps.use_recurrent_dropout
      cell_kwargs['dropout_keep_prob'] = hps.recurrent_dropout_prob
    self.cell = cellfn(hps.rnn_size, **cell_kwargs)

    self.sequence_lengths = tf.placeholder(
        dtype=tf.int32, shape=[self.hps.batch_size], name="seqlengths",
    )
    self.input_data = tf.placeholder(
        dtype=tf.float32,
        shape=[hps.batch_size, hps.max_seq_len, hps.nfeats],
        name="input",
    )
    cell_input = self.input_data

    embedding_dat = [
        ('product', hps.product_embedding_size, N_PRODUCTS),
        ('aisle', hps.aisle_embedding_size, N_AISLES),
        ('dept', hps.dept_embedding_size, N_DEPARTMENTS),
    ]
    input_embeddings = []
    for (name, size, n_values) in embedding_dat:
      if size == 0:
        continue
      embeddings = tf.get_variable('{}_embeddings'.format(name),
          [n_values, size])
      idname = '{}_ids'.format(name)
      input_ids = tf.placeholder(
        dtype=tf.int32, shape=[self.hps.batch_size], name=idname)
      setattr(self, idname, input_ids)
      lookuped = tf.nn.embedding_lookup(
          embeddings,
          input_ids,
          max_norm=None, # TODO: experiment with this param
      )
      lookuped = tf.reshape(lookuped, [self.hps.batch_size, 1, size])
      lookuped = tf.tile(lookuped, [1, self.hps.max_seq_len, 1])
      input_embeddings.append(lookuped)
    if input_embeddings:
      cell_input = tf.concat([self.input_data]+input_embeddings, 2)

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
    # mean loss per sequence (averaged over number of sequence elements not 
    # zeroed out by lossmask - no free rides)
    # tf has way too many ways to do division :[
    loss_per_seq = tf.realdiv(loss_per_seq, 
        tf.reduce_sum(self.lossmask, axis=1)
    )
    # Loss on just the last element of each sequence.
    last_order_indices = self.sequence_lengths - 1 
    self.finetune_cost = tf.reduce_mean(
        tf.gather(loss, last_order_indices)
    )
    # TODO: does the fact that the mean here includes a variable number of dummy
    # values (from padding to max seq len) change anything? Need to be a bit careful.
    # Maybe need to do it in 2 steps? inner avgs. then outer avg-of-avgs?
    # Or just weight by seqlens.
    
    self.cost = tf.reduce_mean(loss_per_seq)
    self.total_cost = self.cost
    #self.cost = tf.reduce_mean(loss)
    if 0 and self.hps.product_embeddings:
      self.weight_penalty = self.hps.embedding_l2_cost * tf.nn.l2_loss(product_embeddings)
      self.total_cost = tf.add(self.cost, self.weight_penalty)
    if self.hps.l2_weight:
      tvs = tf.trainable_variables()
      # Penalize everything except for biases
      l2able_vars = [v for v in tvs if ('bias' not in v.name and 'output_b' not in v.name)]
      self.weight_penalty = tf.add_n([
          tf.nn.l2_loss(v) 
          for v in l2able_vars]) * self.hps.l2_weight
      self.total_cost = tf.add(self.cost, self.weight_penalty)
    else:
      self.weight_penalty = tf.constant(0)

    if self.hps.is_training:
        self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)
        if self.hps.grad_clip:
          gvs = optimizer.compute_gradients(self.total_cost)
          g = self.hps.grad_clip
          capped_gvs = [ (tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
          self.train_op = optimizer.apply_gradients(
              capped_gvs, global_step=self.global_step
          )
        else:
          self.train_op = optimizer.minimize(
                  self.total_cost,
                  self.global_step,
          )
