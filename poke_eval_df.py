import numpy as np
import tensorflow as tf
import pandas as pd

import predictor as pred
import rnnmodel
import utils
import evaluator
from batch_helpers import iterate_wrapped_users, UserWrapper
from insta_pb2 import User

def _get_df(user, predictor):
  prods = user.all_pids
  logits = np.zeros( [user.seqlen, len(prods)] ).T
  pids = list(prods)
  for i, pid in enumerate(pids):
    ts = user.training_sequence_for_pid(pid, predictor.model.hps.max_seq_len)
    gits = predictor._get_logits(ts['x'], ts['seqlen'], ts['pindex'])
    logits[i] = gits[:user.seqlen]

  return pd.DataFrame(logits.T, columns=pids)



def get_eval_df(checkpoint_path="checkpoints/jul1", config='jul1.json'):
  hps = rnnmodel.get_toy_hparams()
  if config:
    with open(config) as f:
      hps.parse_json(f.read())
  hps.is_training = False
  hps.batch_size = 1
  tf.logging.info('Creating model')
  model = rnnmodel.RNNModel(hps)
  
  sess = tf.InteractiveSession()
  # Load pretrained weights
  tf.logging.info('Loading weights')
  utils.load_checkpoint(sess, checkpoint_path)

  tf.logging.info('Loading test set')
  user_pb = User()
  with open('testuser.pb') as f:
    user_pb.ParseFromString(f.read())
  user = UserWrapper(user_pb)

  predictor = pred.RnnModelPredictor(sess, model, .2, predict_nones=0)
  df = _get_df(user, predictor)
  return df

