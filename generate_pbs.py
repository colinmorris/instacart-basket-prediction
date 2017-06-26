import pickle
import tensorflow as tf
import numpy as np
import pandas as pd

from basket_db import BasketDB
from insta_pb2 import User, Order

USER_LIMIT = 1000
MINI = 0
LOAD_CACHED = 1

def main():
  db = BasketDB.load()
  print "Loaded db"
  orders = db.orders
  ops = db.ops
  if MINI:
    orders = orders.head(MINI)
    ops = ops.head(MINI)

  if LOAD_CACHED:
    uid_to_oids = pd.read_pickle('uid_to_oids.pickle')
    oid_to_pids = pd.read_pickle('oid_to_pids.pickle')
  else:
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
    for oid, orow in ordered_orders:
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
    i += 1
    if i >= USER_LIMIT:
      break

  writer.close()

if __name__ == '__main__':
  main()
