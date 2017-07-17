"""This code roughly based on magenta.models.sketch-rnn.model"""
from __future__ import absolute_import, division

import numpy as np
import tensorflow as tf

import rnn
from constants import N_PRODUCTS, N_AISLES, N_DEPARTMENTS

class RNNModel(object):

  def __init__(self, hyperparams, reuse=False):
    self.hps = hyperparams
    self.summaries = []
    with tf.variable_scope('instarnn', reuse=reuse):
      self.build_model()

  def build_model(self):
    hps = self.hps
    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    cell_kwargs = dict(forget_bias=1.0)
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
    elif hps.cell == 'peephole':
      cellfn = tf.nn.rnn_cell.LSTMCell
      cell_kwargs['use_peepholes'] = True

    else:
      assert False, 'please choose a *respectable* cell type'
    if hps.cell not in ('basiclstm', 'peephole'):
      cell_kwargs['use_recurrent_dropout'] = hps.use_recurrent_dropout
      cell_kwargs['dropout_keep_prob'] = hps.recurrent_dropout_prob
    self.cell = cellfn(hps.rnn_size, **cell_kwargs)
    if hps.cell in ('basiclstm', 'peephole'):
      self.cell = tf.nn.rnn_cell.DropoutWrapper(
          self.cell, state_keep_prob=hps.recurrent_dropout_prob)

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
        tf.logging.info('Skipping embeddings for {}'.format(name))
        continue
      embeddings = tf.get_variable('{}_embeddings'.format(name),
          [n_values, size])
      self.add_summary(
          'Embeddings/{}_norm'.format(name), tf.norm(embeddings, axis=1)
          )
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
    # TODO: would like to log forgettitude, but this seems quite tricky. :(

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.hps.rnn_size, 1])
      output_b = tf.get_variable('output_b', [1])
    
    output = tf.reshape(output, [-1, hps.rnn_size])
    logits = tf.nn.xw_plus_b(output, output_w, output_b)
    logits = tf.reshape(logits, label_shape)
    self.logits = logits
    # The logits that were actually relevant to prediction/loss
    boolmask = tf.cast(self.lossmask, tf.bool)
    used_logits = tf.boolean_mask(self.logits, boolmask)
    self.add_summary( 'Logits', used_logits )
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
    r = tf.range(self.hps.batch_size)
    finetune_indices = tf.stack([r, last_order_indices], axis=1)
    self.finetune_cost = tf.reduce_mean(
        tf.gather_nd(loss, finetune_indices)
    )
    
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
        if self.hps.optimizer == 'Adam':
          optimizer_fn = tf.train.AdamOptimizer
        elif self.hps.optimizer == 'LazyAdam':
          optimizer_fn = tf.contrib.opt.LazyAdamOptimizer
        else:
          assert False, "Don't know about {} optimizer".format(self.hps.optimizer)
        self.optimizer = optimizer_fn(self.lr)
        optimizer = self.optimizer
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
        # Log the size of gradient updates to tensorboard
        gradient_varnames = ['RNN/output_w', 'dept_embeddings', 'aisle_embeddings']
        # TODO: how to handle product embeddings? Updates should be v. sparse.
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          gradient_vars = [tf.get_variable(vname) for vname in gradient_varnames]
          # I think compute_gradients shares its implementation with the base Optimizer
          # (i.e. these are the 'raw' gradients, and not the actual updates computed
          # by Adam with momentum etc.)
          grads = optimizer.compute_gradients(self.total_cost, var_list=gradient_vars)
          for (grad, var) in grads:
            colidx = var.name.rfind(':')
            basename = var.name[ len('instarnn/') : colidx ]
            summ_name ='Gradients/{}'.format(basename)
            self.add_summary(summ_name, grad)

          if self.hps.cell != 'lstm':
            tf.logging.warn("Histogram logging not implemented for cell {}".format(self.hps.cell))
            return
          cellvars = [tf.get_variable('rnn/LSTMCell/'+v) 
            for v in ['W_xh', 'W_hh', 'bias']
            ]
          # TODO: Ideally we could go even further and cut up the xh gradients by
          # feature (family). That'd be siiiiiiiick.
          cellgrads = optimizer.compute_gradients(self.total_cost, var_list=cellvars)
          for (grad, var) in cellgrads:
            colidx = var.name.rfind(':')
            parenidx = var.name.rfind('/')
            basename = var.name[parenidx+1:colidx]
            bygate = tf.split(grad, 4, axis=0 if basename == 'bias' else 1)
            gates = ['input_gate', 'newh_gate', 'forget_gate', 'output_gate']
            for (subgrads, gatename) in zip(bygate, gates):
              summname = 'Gradients/LSTMCell/{}/{}'.format(basename, gatename)
              self.add_summary(summname, subgrads)



  def add_summary(self, name, tensor):
    # Without this, all histogram summaries get plopped under the same "instarn"
    # tag group.
    with tf.name_scope(None):
      summ = tf.summary.histogram(name, tensor)
      self.summaries.append(summ)

  def merged_summary(self):
    return tf.summary.merge(self.summaries)

