import time
from basket_db import BasketDB
  
t0 = time.time()
db = BasketDB.load()
orders = db.orders
ops = db.ops
uid_to_oids = orders.groupby('user_id')['order_id'].apply(list)
# Y'know, I think these might already be sorted incidentally? Cause that's
# how they're ordered in the files
oid_to_pids = ops.sort_values(['order_id', 'add_to_cart_order'])\
    .groupby('order_id')['product_id'].apply(list)

uid_to_oids.to_pickle('uid_to_oids.pickle')
oid_to_pids.to_pickle('oid_to_pids.pickle')

elapsed = time.time() - t0
print "Finished in {:.1f}s".format(elapsed)
