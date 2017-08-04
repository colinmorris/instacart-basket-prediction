#!/usr/bin/env python
from __future__ import division
import argparse
import pickle
from collections import defaultdict

from baskets import common
from baskets.user_wrapper import iterate_wrapped_users
from baskets.time_me import time_me

from feature_spec import FeatureSpec

# "raw" generic (applying to whole order) and product-specific features
# (These variables aren't actually used anywhere. Just here for reference.)
generic_feats = ['hour', 'dow', 'days_since_prior', 'uid', ]
# recency_days may be 0 if the product was in an order from earlier in the same
# day. Otherwise, everything except label is always positive.
pidfeats = ['pid', 'frequency', 'recency_days', 'recency_orders', 'label']

class Vectorizer(object):

  def __init__(self, featspec):
    self.featspec = featspec
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
    return i

  def write_example(self, example):
    # Not clear whether libfm actually requires feats to be sorted.
    featdict = self.featspec.make_featdict(example)
    def stringify_val(val):
      if isinstance(val, float):
        return '{:.4g}'.format(val)
      return str(val)
    # TODO: should see if sorting here is a bottleneck. If it is, pretty easy
    # to have featspec return an OrderedDict. (Keys are already added in almost
    # sorted order.)
    # (After profiling, looks like this line is the biggest time sink, but 
    # the cost of sorting is relatively small compared to the calls to str.format)
    feat_str = ' '.join(
        '{}:{}'.format(i, '{:.4g}'.format(featdict[i]))
        for i in sorted(featdict)
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
    group_fname = 'groups.txt'
    with open(group_fname, 'w') as f:
      self.featspec.write_group_file(f)

  def save_examples(self):
    self.train_out.close()
    self.test_out.close()

class Example(object):

  def __init__(self, pid, gfs, pfs, test):
    self.pid = pid
    # Feature name -> value
    self.gfs = gfs
    # pid -> feature name -> value
    self.pfs = pfs
    self.label = self.pfs[pid]['label']
    self.test = test

def get_order_dat(user):
  """Yield features for each order in the user's history eligible to be a training
  instance (i.e. all of them except the first, and testorder if present)"""
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
    # XXX: A tricky thing to be aware of here is that we mutate pfs
    # after we yield it, so there's a risk that a receiver of these (e.g. 
    # Example, which uses these as attribute values) may have them change under their
    # feat. (Not a problem with current control flow, but something to be aware of.)
    # Also, if, say, an Example mutates one of these dicts, it'll affect all the other
    # Examples that share a pointer to it. Would be cool if there was a way to 
    # yield immutable copies to not have to worry about this stuff.
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
  # TODO: would be kind of nice to have an optional tag to be able to juggle
  # between sets of vectors corresponding to different user folds or sets
  # of features. (Though don't want too many lying around. Vectorizing just
  # 1% of users can already produce files as big as a GB.)
  parser = argparse.ArgumentParser()
  parser.add_argument('user_fold')
  parser.add_argument('--lim', type=int, help='Limit number of users vectorized')
  args = parser.parse_args()
  
  #featspec = FeatureSpec.all_features_spec()
  featspec = FeatureSpec.basic_spec()
  victor = Vectorizer(featspec)
  users = iterate_wrapped_users(args.user_fold)
  n = victor.vectorize_users(users, limit=args.lim)
  print 'Vectorized {} users from fold {}'.format(n, args.user_fold)

if __name__ == '__main__':
  with time_me(mode='print'):
    main()
