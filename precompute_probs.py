import os
import argparse
import time
import pickle
import tensorflow as tf
import numpy as np
from scipy.special import expit
from collections import defaultdict

import rnnmodel
import model_helpers
import utils
import batch_helpers as bh

def get_probmap(batcher, model, sess, userlimit):
  # {uid -> {pid -> prob}}
  pmap = defaultdict(dict)
  batches = batcher.get_batches(pids_per_user=-1, infinite=False,
      allow_smaller_final_batch=True)
  bs = model.hps.batch_size
  i = 0
  uids_seen = set()
  for batch in batches:
    _x, _labels, seqlens, _lm, pindexs, uids = batch
    feed = model_helpers.feed_dict_for_batch(batch, model)
    logits = sess.run(model.logits, feed)
    assert logits.shape == (bs, model.hps.max_seq_len)
    final_logits = logits[np.arange(bs),seqlens-1]
    final_probs = expit(final_logits)
    skipped = 0
    for uid, pindex, prob in zip(uids, pindexs, final_probs):
      if uid == pindex == 0:
        skipped += 1
        continue
      pid = pindex+1
      pmap[uid][pid] = prob
    if skipped:
      print "Skipped {} elements of batch".format(skipped)

    i += 1
    uids_seen.update(set(uids))
    if userlimit and len(uids_seen) >= userlimit:
      break
  print "Computed probabilities for {} users in {} batches".format(len(pmap), i)
  return pmap

def main():
  parser = argparse.ArgumentParser()
  # TODO: refactor this to just take a tag, like eval.py
  parser.add_argument('checkpoint_path')
  parser.add_argument('--recordfile', default='test.tfrecords', 
      help='tfrecords file with the users to test on (default: test.tfrecords)')
  parser.add_argument('-n', '--n-users', type=int, 
      help='Limit number of users tested on (default: no limit)')
  parser.add_argument('-c', '--config', default=None,
      help='json file with hyperparam overwrites') 
  parser.add_argument('--dest',
      help='destination path for pickled dict of uid -> pid -> prob '
      '(default: pdicts/[tag].pickle where [tag] is the name of the last dir in checkpoint_path)')
  args = parser.parse_args()

  hps = rnnmodel.get_toy_hparams()
  if args.config:
    with open(args.config) as f:
      hps.parse_json(f.read())
  hps.is_training = False
  tf.logging.info('Creating model')
  model = rnnmodel.RNNModel(hps)
  
  sess = tf.InteractiveSession()
  # Load pretrained weights
  tf.logging.info('Loading weights')
  utils.load_checkpoint(sess, args.checkpoint_path)
  batcher = bh.Batcher(hps, args.recordfile)

  t0 = time.time()
  probmap = get_probmap(batcher, model, sess, args.n_users)
  dest = args.dest
  if dest is None:
    head, tail = os.path.split( os.path.normpath(args.checkpoint_path) ) 
    tag = tail
    dest = 'pdicts/{}.pickle'.format(tag)
  with open(dest, 'w') as f:
    pickle.dump(dict(probmap), f)
  elapsed = time.time() - t0
  print "Wrote probability dict to {}. Finished in {:.1f}s".format(
      dest, elapsed
      )

if __name__ == '__main__':
  main()
