import logging
import random
import numpy as np
import tensorflow as tf

from baskets.insta_pb2 import User
from baskets import features
from baskets.features import FEATURES, NFEATS
from baskets.constants import NONE_PRODUCTID
from baskets import utils
from baskets import common
from baskets.user_wrapper import UserWrapper

class Batcher(object):
  def __init__(self, hps, recordpath, in_media_res=False, testmode=False, finetune=False):
    """Test mode => we're making predictions on the (Kaggle-defined) test set.
    The vectors we output should include the users' final orders (for which
    we don't know the ground truth labels)."""
    self.hps = hps
    self.recordpath = common.resolve_recordpath(recordpath)
    self.batch_size = hps.batch_size
    self.nfeats = hps.nfeats
    self.testmode = testmode
    self.finetune = finetune
    if self.nfeats != NFEATS:
      # We're not using all the crayons in the crayon box. Either we've deliberately
      # chosen not to use some features, or we're running in 'legacy mode' (i.e. we've
      # added more features since we trained this model).
      feats = features.lookup_features(hps.feats)
      assert sum(f.arity for f in feats) == hhps.nfeats
      self.feat_fixture = (feats, hps.nfeats)
    else:
      self.feat_fixture = None
    self.max_seq_len = hps.max_seq_len
    if (hps.aisle_embedding_size or hps.dept_embedding_size):
      self.product_df = utils.load_product_df()
    self.reset_record_iterator()
    if in_media_res:
      assert recordpath.endswith('train.tfrecords'), \
        "Don't know how many records in {}".format(recordpath)
      self.random_seek()

  def random_seek(self):
    # TODO: If we wanted to make this more non-deterministic, we could probably
    # do some math to find the exact spot to resume at, based on the restored
    # value of global_step
    nusers = 195795
    nskipped = random.randint(0, nusers)
    for _ in range(nskipped):
      self.records.next()

  def reset_record_iterator(self):
    self.records = tf.python_io.tf_record_iterator(self.recordpath)

  def get_batches(self, pids_per_user=1, infinite=True, allow_smaller_final_batch=False):
    """if pids_per_user == -1, then use all pids
    TODO: should maybe yield dicts rather than massive tuples
    """
    bs = self.batch_size
    maxlen = self.max_seq_len
    # XXX: is it okay to overwrite and re-yield these ndarrays?
    # proooobably? callers should never modify their values, and
    # after they're ready for the next() batch, they should have
    # no reason to be interacting with the previous one. But gotta
    # be careful.
    x = np.zeros([bs, maxlen, self.nfeats])
    labels = np.zeros([bs, maxlen])
    seqlens = np.zeros([bs], dtype=np.int32)
    lossmask = np.zeros([bs, maxlen])
    # (These are actually pids minus one. Kaggle pids start from 1, and we want
    # our indices to start from 0.)
    pids = np.zeros([bs], dtype=np.int32)
    aids = np.zeros([bs], dtype=np.int32)
    dids = np.zeros([bs], dtype=np.int32)
    uids = np.zeros([bs], dtype=np.int32)


    # TODO: Not clear if it's necessary to call this between batches.
    # Ideally we'd be smart enough not to care about any junk beyond :seqlen
    # in any of the returned arrays, but not sure that's the case.
    def _clear_batch_arrays():
      # Only need to do this for arrays with values per sequence element,
      # rather than just per sequence. (Because they have variable length,
      # and we don't want leftover cliffhangers.)
      for arr in [x, labels, lossmask]:
        arr[:] = 0

    i = 0 # index into current batch
    while 1:
      user = User()
      try:
        user.ParseFromString(self.records.next())
      except StopIteration:
        if infinite:
          logging.info("Starting a new 'epoch'. Resetting record iterator.")
          self.reset_record_iterator()
          user.ParseFromString(self.records.next())
        else:
          if allow_smaller_final_batch and i > 0:
            # Set remaining uid/pid/seqlen slots to 0 as signal to caller that
            # these are just dummy/padding values
            for arr in [pids, uids, seqlens]:
              arr[i:] = 0
            yield x, labels, seqlens, lossmask, pids, aids, dids, uids
          # (Not clear if we should do this as a matter of course, or leave it up to caller)
          self.reset_record_iterator()
          raise StopIteration
      # In finetune mode, use only users from the 'train' set (Kaggle-defined)
      if self.finetune and user.test:
        continue
      wrapper = UserWrapper(user, self.feat_fixture)
      if pids_per_user == 1:
        user_pids = [wrapper.sample_pid()]
      elif pids_per_user == -1:
        user_pids = wrapper.all_pids
      else:
        user_pids = wrapper.sample_pids(pids_per_user)
        # This warning is kind of spammy. But yeah, out of 10k users in 
        # validation set, a couple dozen have only one eligible product id.
        if 0 and len(user_pids) < pids_per_user:
          logging.warning('User {} has only {} pids (< {})'.format(
            user.uid, len(user_pids), pids_per_user)
            )

      for pid in user_pids:
        ts = wrapper.training_sequence_for_pid(pid, maxlen, 
            product_df=self.product_df,
            testmode=self.testmode)
        x[i] = ts['x']
        labels[i] = ts['labels']
        seqlens[i] = ts['seqlen']
        lossmask[i] = ts['lossmask']
        if self.finetune:
          final_idx = ts['seqlen'] -1
          assert lossmask[i, final_idx] == 1
          lossmask[i,:] = 0
          lossmask[i,final_idx] = 1
        pids[i] = ts['pindex']
        aids[i] = ts['aisle_id']
        dids[i] = ts['dept_id']
        uids[i] = user.uid
        i += 1
        if i == bs:
          yield x, labels, seqlens, lossmask, pids, aids, dids, uids
          i = 0
          _clear_batch_arrays()
          

def iterate_wrapped_users(recordpath):
  recordpath = common.resolve_recordpath(recordpath)
  records = tf.python_io.tf_record_iterator(recordpath)
  for record in records:
    user = User()
    user.ParseFromString(record)
    yield UserWrapper(user)

class TestBatcher(object):
  def __init__(self, hps):
    self.hps = hps
    u = User()
    with open('testuser.pb') as f:
      u.ParseFromString(f.read())
    self.user = UserWrapper(u)
    self.product_df = utils.load_product_df()

  def batch_for_pid(self, pid):
    ts = self.user.training_sequence_for_pid(pid, self.hps.max_seq_len, self.product_df)
    for k, v in ts.iteritems():
      if isinstance(v, np.ndarray):
        ts[k] = np.expand_dims(v, axis=0)
      elif isinstance(v, int):
        ts[k] = np.array([v])
    return (ts['x'], ts['labels'], ts['seqlen'], ts['lossmask'],
        ts['pindex'], ts['aisle_id'], ts['dept_id'], [self.user.uid])


def vectorize(df, user, maxlen, features=None, nfeats=None):
  features = features or FEATURES # Default to all of them
  nfeats = nfeats or NFEATS
  res = np.zeros([maxlen, nfeats])
  i = 0
  seqlen = len(df) 
  for feat in features:
    featvals = feat.fn(df, user)
    if feat.arity == 1:
      res[:seqlen,i] = featvals
    else:
      res[:seqlen,i:i+feat.arity] = featvals
    i += feat.arity
  return res
