#!/usr/bin/env python

"""Hacky lil script for exploring predictions"""

import argparse
from collections import defaultdict

from baskets import common, utils, predictor as pred
from baskets.user_wrapper import iterate_wrapped_users
from baskets.results import OrderResults

class UserWrapperWrapper(object):
  def __init__(self, uw, prod_lookup):
    self.uw = uw
    self.prod_lookup = prod_lookup

  def translate_pids(self, pids):
    return [
        'none' if pid == -1 else self.prod_lookup.loc[pid-1, 'product_name'] 
        for pid in pids
        ]

  @property
  def lastorder(self):
    return self.translate_pids(self.uw.user.orders[-1].products)

  @property
  def previous_orders(self):
    res = []
    for order in self.uw.user.orders[-1::-1]:
      #res.append( self.translate_pids(order.products) )
      res.append(order)
    return res

  def product_history(self, prod, lim=None):
    prodpid = self.prod_lookup[self.prod_lookup['product_name']==prod]\
        ['product_id'].iloc[0]
    since = self.uw.user.orders[-1].days_since_prior
    daymarker = '.'
    s = ''
    for i, prev in enumerate(self.previous_orders):
      if lim and i >= lim:
        break
      s += daymarker * since
      s += 'Y' if prodpid in prev.products else 'n'
      since = prev.days_since_prior
    #print s
    return s

  def mostwanted(self, n=10):
    c = defaultdict(int)
    for order in self.previous_orders:
      prods = self.translate_pids(order.products)
      for prod in prods:
        c[prod] += 1
    items = c.items()
    items.sort(key=lambda tup:tup[1], reverse=True)
    return items[:n]

  def print_prevs(self, lim=None):
    running_days = self.uw.user.orders[-1].days_since_prior
    for i, prev in enumerate(self.previous_orders):
      if lim and i >= lim:
        break
      print '{} days ago... {}'.format(running_days, 
          self.translate_pids(prev.products))
      running_days += prev.days_since_prior


def foo(user, predictor, pl):
  def lookup(pid):
    return pl.loc[pid-1 ,'product_name']
  # TODO: interesting to know threshold
  uww = UserWrapperWrapper(user, pl)
  actual_pids = user.last_order_predictable_prods()
  actual = uww.translate_pids(actual_pids) 
  predicted_pids = predictor.predict_last_order(user)
  predicted = uww.translate_pids(predicted_pids)
  pid_to_prob = predictor.probmap[user.uid]
  prod_to_prob = {lookup(pid): prob for (pid, prob) in pid_to_prob.iteritems()}
  sorted_prods = sorted(prod_to_prob.items(), key=lambda tup: tup[1], reverse=True)
  res = OrderResults.for_pids(predicted_pids, actual_pids)
  print_prevs = uww.print_prevs
  ph = uww.product_history
  print "On to user {} with {} orders".format(user.uid, user.norders)
  #ph('Large Lemon', 10)
  import pdb; pdb.set_trace()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('tag')
  parser.add_argument('--recordfile', default='test', 
      help='identifier for user stratum')
  args = parser.parse_args()

  product_lookup = utils.load_product_df()

  pmap = common.pdict_for_tag(args.tag, args.recordfile)
  user_iterator = iterate_wrapped_users(args.recordfile)
  predictor = pred.HybridThresholdPredictor(pmap, optimization_level=0)

  skip = 10
  for i, user in enumerate(user_iterator):
    if i < skip:
      continue
    foo(user, predictor, product_lookup)

if __name__ == '__main__':
  main()
