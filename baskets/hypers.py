import os
import logging

from tensorflow.contrib.training import HParams

from baskets.features import NFEATS, FEATURES
from baskets import common

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
      # 256 seems like the sweet spot for this. More powerful than 128.
      # 512 is veeeery slow, and seems to have bad convergence
      rnn_size=256,
      # Haven't experimented with this yet. Larger size might be nice for smoothing 
      # out gradient updates? Especially in 'finetune' mode, where training signal
      # is very sparse/noisy compared to normal.
      batch_size=100,
      # TODO: Should really be doing some kind of dynamic padding - where we
      # pad each batch to the length of its longest sequence.
      max_seq_len=100,
      # More correctly, the dimensionality of the feature space (there are 
      # some "features" that correspond to 2+ numbers, e.g. onehot day of week
      nfeats=NFEATS,
      feats=[f.name for f in FEATURES],
      # Note that the learning rate used for training is a fn of the initial
      # value, the decay/min, and the current *global_step*. So if you start
      # train from an existing checkpoint, the learning rate will not start
      # at the value below. Unless you change the param under it.

      # The current run to beat uses an initial lr of 0.01 with a decay of .9995 or .9998
      # (May want to update defaults to match this).
      # Seems like a highly impactful hyperparameter, and also a sensitive one.
      # It's possible that other variants that have been unsuccessful so far 
      # (e.g. different RNN cells, batch_norm) have only been so because they 
      # perform best at a different learning_rate compared to the vanilla config.
      # (So far, have mostly been tweaking hyperparams in isolation. If I had
      # a faster computer, it'd probably make sense to do a random search,
      # varying all hps simultaneously.)
      learning_rate=0.001,
      # If True, start learning rate schedule from the above number, and calculate
      # the decay wrt steps taken in the current run (and *not* global_step). Only
      # affects runs resumed from a checkpoint.
      lr_reset=False, 
      # Current runs to beat use around .9995-.9998. Set to 1 to disable lr decay.
      decay_rate=0.9999, 
      # Current run to beat uses .0001, but this is probably too high.
      min_learning_rate=0.00001,
      save_every=5000,
      # TODO: I'd kind of like to be able to do eval a little more frequently at
      # the beginning, because loss is changing more then? (Also, to be clear,
      # this is how often to measure loss on *validation* data, not eval)
      eval_every=2000,
      # How often to log training loss and some other stuff. Unlike measuring
      # the validation loss, there's no cost to doing this more often (it's
      # reporting numbers that we get for free during training). But logging
      # with frequency <= ~200 leads to some unpleasantly spiky graphs in
      # Tensorboard. (I know it has smoothing, but I prefer that to be off when
      # looking at eval_cost, and you can't tune it per variable. :()
      # (Actually, it's no longer true that this has no additional cost, since
      # we now fetch histogram summaries at this interval)
      log_every=500,
      # There are about 195k users in the dataset, so if we take one sequence
      # from each, it'd take about 2k steps to cycle through them all (with batch_size=100). 
      # But the average user has around 65 eligible products, so the number
      # of batches to match the full size of the dataset is more like 130k.
      # (Though, because of how sampling is done, probably at least like
      # 75% of the data would still be unseen by the model after that many steps.)
      num_steps=10000,
      product_embedding_size=32,
      # Embeddings for aisle and department (22 depts, 135 aisles in dataset)
      # Set to 0 to disable these embeddings.
      aisle_embedding_size=8,
      # TODO: Since (afair) there's a 1:1 mapping from aisle to dept, and there
      # are only 135 distinct aisles, maybe dept embeddings are overkill.
      dept_embedding_size=4,
      # gradient clipping. Set to falsy value to disable. Experiments so far
      # have been unsuccessful. Might end up being important for fine-tuning
      # (because of sparse training signal, more potential for vanishing/
      # exploding gradients?)
      grad_clip=0.0, 
      # Scaling factor for L2 penalty applied to all trainable weights (save biases).
      l2_weight=.00001,
      # Dropout. This seems to help a fair bit!
      use_recurrent_dropout=True,
      # XXX: this is the *keep* prob
      # .9 seems like a pretty sweet spot. Have experimented with 
      # larger/smaller values, and got worse results.
      recurrent_dropout_prob=.9,
      # One of {lstm, layer_norm, hyper} (all from rnn.py) or one of 
      # {basiclstm, peephole} (from tf.nn.rnn_cell). So far have had
      # bad luck with everything but lstm.
      #   - hyper and layer_norm gave mediocre performance and were VERY
      #     slow (like 4x slower).
      #  - peephole was mediocre, but at least no slower. (Part of the reason
      #    for its worseness might be lack of ortho init)
      cell='lstm', 
      # One of {Adam, LazyAdam}
      optimizer='Adam',

      fully_specified=False, # Used for config file bookkeeping
  )

def hps_for_tag(tag, save_full=False):
  hps = get_default_hparams()
  config_path = '{}/{}.json'.format(common.CONFIG_DIR, tag)
  with open(config_path) as f:
    hps.parse_json(f.read())
  if save_full:
    full_hps = copy_hps(hps)
    full_hps.fully_specified = True
    basename = '{}_{}.json'.format(tag, int(time.time()))
    full_path = os.path.join(common.CONFIG_DIR, 'full', basename)
    with open(full_path, 'w') as f:
      f.write(hps.to_json())
  return hps

def as_eval(hps):
  eval_hps = copy_hps(hps)
  eval_hps.is_training = False
  eval_hps.use_recurrent_dropout = False
  return eval_hps


def copy_hps(hps):
  return HParams(**hps.values())
