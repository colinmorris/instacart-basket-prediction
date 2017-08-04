#!/usr/bin/env python
from __future__ import division
import argparse
import pickle
import random
import math
from collections import defaultdict

from baskets import common, constants
from baskets.user_wrapper import iterate_wrapped_users
from baskets.time_me import time_me

generic_feats = ['hour', 'dow', 'days_since_prior', 'uid', ]

# recency_days may be 0 if the product was in an order from earlier in the same
# day. Otherwise, everything except label is always positive.
pidfeats = ['pid', 'frequency', 'recency_days', 'recency_orders', 'label']

# TODO: would be nice to be able to customize certain pid feats to be included
# for the focal product but not the others.
a_config = dict(
    generic_feats = generic_feats,
    focal_pid_feats = [pf for pf in pidfeats if pf != 'label'],
    pid_feats = [], #['presence'],
)

# Consistency is key here. Need to be able to vectorize multiple tfrecord files
# and get the same feature<->id mapping. 
# (also need to be able to manage multiple configurations of features)
# But maybe we can worry about this later...
class Vectorizer(object):

  def __init__(self, fold):
    self.fold = fold
    self.config = a_config
    self.featurizer = Featurizer(self.config)
    self._test_ids = []

  def vectorize_users(self, users, limit=None):
    train_path = common.resolve_libfm_vector_path('train')
    test_path = common.resolve_libfm_vector_path('test')
    self.train_out = open(train_path, 'w')
    self.test_out = open(test_path, 'w')
    i = 0
    for user in users:
      for example in make_examples(user):
        self.write_example(example)
      i += 1
      if limit and i >= limit:
        break
    self.finish()

  def write_example(self, example):
    # Does libfm require features be sorted? idk. couldn't hurt I guess.
    featdict = self.featurizer.featurize(example)
    def stringify_val(val):
      if isinstance(val, float):
        return '{:.4g}'.format(val)
      return str(val)
    feat_str = ' '.join(
        '{}:{}'.format(i, stringify_val(featdict[i]))
        for i in sorted(featdict.keys())
        )
    line = '{} {}\n'.format(example.label, feat_str)
    out = self.test_out if example.test else self.train_out
    out.write(line)
    if example.test:
      ids = [example.gfs['uid'], example.pid]
      self._test_ids.append(ids)

  def finish(self):
    self.save_examples()
    with open('test_ids.pickle', 'w') as f:
      pickle.dump(self._test_ids, f)
    # save groups

  def save_examples(self):
    self.train_out.close()
    self.test_out.close()

  def save_groups(self):
    """For grouping, libfm wants a text file with as many lines as features in the
    dataset, where the ith line gets k, where feature_i \in group_k
    """
    pass


class Example(object):

  def __init__(self, pid, gfs, pfs, test):
    self.pid = pid
    # Feature name -> value
    self.gfs = gfs.copy()
    # pid -> feature name -> value
    self.pfs = pfs
    self.label = self.pfs[pid]['label']
    self.test = test

    # XXX: Feature transformation hacks
    for pf in self.pfs.itervalues():
      # ridiculous hacks
      pf['frequency_'] = math.log(pf['frequency'])
      pf['recency_days_'] = 4 / (4 + pf['recency_days'])
      pf['recency_orders_'] = 2 / (2 + pf['recency_orders'])

class Featurizer(object):
  GROUP_SIZES = dict(
      uid = constants.N_USERS,
      hour = 24,
      dow = 7,
      days_since_prior = 1
  )
  ONE_INDEXED_GROUPS = {'uid'}
  def __init__(self, config):
    self.config = config
    self.featmap = {gf: i for i, gf in enumerate(config['generic_feats'])}

  def get_offset(self, group):
    offset = 0
    for groupx in self.config['generic_feats']:
      if groupx == group:
        return offset
      offset += self.GROUP_SIZES[groupx]
    assert group == -1
    return offset

  def featurize(self, example):
    # TODO: Maybe a lot of this logic should move to Example
    f = {}
    # Generic scalar feats
    for gf in self.config['generic_feats']:
      groupsize = self.GROUP_SIZES[gf]
      if groupsize == 1:
        i = self.get_offset(gf)
        val = example.gfs[gf]
      else:
        i = self.get_offset(gf) + example.gfs[gf]
        if gf in self.ONE_INDEXED_GROUPS:
          i -= 1
        val = 1
      f[i] = val

    offset = self.get_offset(-1)
    for j, fpf in enumerate(self.config['focal_pid_feats']):
      # haaacks
      if fpf == 'pid':
        i = offset + (example.pid - 1)
        f[i] = 1
        offset += constants.N_PRODUCTS
      else:
        pfdict = example.pfs[example.pid]
        try:
          f[offset] = pfdict[fpf+'_'] # XXX: Haaaaaaaaaaack
        except KeyError:
          f[offset] = pfdict[fpf]
        offset += 1

    #offset += len(self.config['focal_pid_feats'])
    for pf in self.config['pid_feats']:
      for pid in example.pfs:
        if pid == example.pid:
          continue
        i = offset + (pid-1)
        # hackesque
        if pf == 'presence':
          val = 1
        else:
          val = example.pfs[pid][pf]
        f[i] = val
      offset += constants.N_PRODUCTS

    return f

def get_order_dat(user):
  # product feats
  pfs = defaultdict(lambda : defaultdict(int))
  for i, (prev_order, order) in enumerate(user.order_pairs()):
    # generic feats
    gfs = dict(uid=user.uid, hour=order.hour, dow=order.dow, 
        days_since_prior=order.days_since_prior,
        )
    for (cart_index, pid) in enumerate(prev_order.products):
      pf = pfs[pid]
      pf['frequency'] += 1
      pf['recency_days'] = 0
      pf['recency_orders'] = 0
      pf['label'] = 2 # ha ha hacks
    # tock
    for pf in pfs.itervalues():
      pf['recency_days'] += order.days_since_prior
      pf['recency_orders'] += 1
      pf['label'] = max(0, pf['label']-1)
    yield gfs, pfs

def make_examples(user):
  # TODO: may want to experiment with taking less context. Or maybe not. Tradeoff
  # between wanting fidelity to the test distribution and wanting to give the model
  # as much information as possible.
  max_i = len(user.user.orders)-2
  for i, (gfs, pfs) in enumerate(get_order_dat(user)):
    for pid in pfs:
      yield Example(pid, gfs, pfs, test=i==max_i)

def main():
  random.seed(1337)
  parser = argparse.ArgumentParser()
  parser.add_argument('user_fold')
  args = parser.parse_args()

  victor = Vectorizer(args.user_fold)
  users = iterate_wrapped_users(args.user_fold)
  victor.vectorize_users(users) #, limit=1000) # XXX

if __name__ == '__main__':
  with time_me(mode='print'):
    main()
