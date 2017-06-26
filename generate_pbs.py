import pickle
import time
import tensorflow as tf
import numpy as np
import pandas as pd

from basket_db import BasketDB
from insta_pb2 import User, Order

USER_LIMIT = 0
LOAD_CACHED = 1
TEST_UID = 2455

def main():
  LOG_EVERY = 5000
  db = BasketDB.load(dropops=LOAD_CACHED)
  print "Loaded db"
  orders = db.orders
  # TODO: minor(?) memory optimization: don't need to load ops

  if LOAD_CACHED:
    uid_to_oids = pd.read_pickle('uid_to_oids.pickle')
    oid_to_pids = pd.read_pickle('oid_to_pids.pickle')
  else:
    ops = db.ops
    uid_to_oids = orders.groupby('user_id')['order_id'].apply(list)
    oid_to_pids = ops.groupby('order_id')['product_id'].apply(list)
  print "Done lookups"

  writer = tf.python_io.TFRecordWriter('users.tfrecords')
  i = 0
  for uid, oids in uid_to_oids.iteritems():
    user = User()
    user.uid = uid
    # Orders ordered chronologically. Fuck the English language btw.
    ordered_orders = orders.loc[oids].sort_values('order_number')
    for oid, orow in ordered_orders.iterrows():
      if orow.eval_set == 'test':
        user.test = True
        # Skip any orders in the test set, since we don't have 
        # ground truth labels for them.
        # TODO: Maybe these should be written to a different file?
        continue
      order = user.orders.add()
      order.orderid = oid
      order.nth = orow.order_number
      order.dow = orow.order_dow
      order.hour = orow.order_hour_of_day
      days = orow.days_since_prior_order
      if not pd.isnull(days):
        order.days_since_prior = int(days)
      try:
        order.products.extend(oid_to_pids.loc[oid])
      except KeyError:
        assert MINI

    writer.write(user.SerializeToString())
    if uid == TEST_UID:
      print "Writing uid {} to testuser.pb".format(uid)
      with open('testuser.pb', 'w') as f:
        f.write(user.SerializeToString())
    i += 1
    if USER_LIMIT and i >= USER_LIMIT:
      break
    if i % LOG_EVERY == 0:
      print "{} users written".format(i)

  writer.close()

if __name__ == '__main__':
  t0 = time.time()
  main()
  t1 = time.time()
  print "Finished in {:.1f}s".format(t1-t0)
