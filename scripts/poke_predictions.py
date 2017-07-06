import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import pandas as pd
import numpy as np

import batch_helpers as bh

fname = sys.argv[1]
with open(fname) as f:
  predmap = pickle.load(f)

recordfname = 'test.tfrecords'

threshes = []
cols = ['pid', 'nprev', 'correct', 'orders_since']
rows = []

def make_row(user, pid):
  orders = user.user.orders
  correct = pid in orders[-1].products
  prev_orders = [i for (i, order) in enumerate(orders[:-1]) if pid in order.products]
  nprev = len(prev_orders)
  orders_since = (user.norders-1) - prev_orders[-1]
  return [pid, nprev, correct, orders_since]
for user in bh.iterate_wrapped_users(recordfname):
  pred = predmap[user.uid]
  threshes.append(pred['thresh'])
  for pid in pred['predicted']:
    if pid == -1:
      continue
    row = make_row(user, pid)
    rows.append(row)

df = pd.DataFrame(rows, columns=cols)
