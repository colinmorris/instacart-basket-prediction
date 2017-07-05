import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
import rnnmodel
import utils
import model_helpers as helpers

try:
  tag = sys.argv[1]
except IndexError:
  print "Pass in a tag, dummy"
  sys.exit(1)

ckpt = 'checkpoints/{}'.format(tag)

hps = helpers.hps_for_tag(tag)
hps.is_training = 0
hps.batch_size = 1
model = rnnmodel.RNNModel(hps)
sess = tf.InteractiveSession()
utils.load_checkpoint(sess, ckpt)

def lookup(varname):
  with tf.variable_scope('instarnn', reuse=True):
      var = tf.get_variable(varname)
  val = sess.run(var)
  return val

emb = lookup('product_embeddings')
# Based on my reading of the tf code, the weights at the start are
# for the cell inputs, and are followed by the weights on the previous
# state.
# In our case, the first ~20 (as of jul2 model) are the feature-features,
# followed by product embedding weigths (32-dim embeddings in jul2), then
# 128 weights on the prev state.
lstm_weights = lookup('rnn/basic_lstm_cell/weights')
print emb.shape

#emb[:10]

