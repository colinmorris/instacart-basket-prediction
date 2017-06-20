import pandas as pd
import numpy as np
import logging
import argparse
import time

#import cProfile

from basket_db import BasketDB

class OrderSequenceVectorizer(object):

  def __init__(self, db):
    self.db = db

  #@profile
  def ops_for_orders(self, orders):
    """returns ops df. input is a subset of orders df"""
    oids = orders['order_id']
    return self.db.ops.loc[oids]

  def order_features(self, orders):
    vecs = []
    for (_, order) in orders.iterrows():
      # TODO: bleh
      break
      # soooo... all of them
      cols = ['order_id',
          'user_id', 'eval_set', 'order_number', 'order_dow', 
           'order_hour_of_day', 'days_since_prior_order']
    return orders.values.tolist()

  def ops_features(self, ops, oids):
    # TODO: this could be rewritten now to use multi-index
    by_order = ops.groupby(level='order_id')
    order_feats = {}
    for (oid, group) in by_order:
      # TODO: not counting focal product if present?
      # number of products in this order
      nprods = len(group)
      n_reorders = group['reordered'].sum()
      # TODO: is add_to_cart_order of reordered items particularly
      # useful/interesting?
      feats = [nprods, n_reorders]
      order_feats[oid] = feats

    return [order_feats[id] for id in oids]

  def get_vector_df(self, limit=None):
    vlists = self.vector_lists(limit)
    cols = (
        ['seqid', 'prodid', 'reordered',]
        + self.db.orders.columns.tolist()
        + ['prev_prods', 'prev_reorders']
        )
    df = pd.DataFrame(vlists, columns=cols)
    return df
        

  #@profile
  def vector_lists(self, limit=None):
    orders = self.db.orders
    # TODO: maybe could just speed whole thing up by doing an initial join
    # between orders and ops, and keeping most of the rest of the code the same
    orders_by_user = orders.groupby('user_id')
    # TODO: maybe should be an ndarray? 
    # but would need uniform type?
    # TODO: memory of this might be crazy. May need to build incrementally. Ugh.
    rows = []
    next_seqid = 0
    log_every = 1000
    nextlog = log_every
    t0 = time.time()
    users = 0
    for (uid, group) in orders_by_user:
      # I think this is redundant but whatever
      group = group.sort_values('order_number')
      norders = len(group)

      # ordered product feats are lagged one order behind the 
      # order features
      ops = self.ops_for_orders(group) # .iloc[:-1] ?
      
      order_feats = self.order_features(group.iloc[1:])
      lagged_oids = group['order_id'].iloc[:-1]
      ops_feats = self.ops_features(ops,  lagged_oids)
      assert len(ops_feats) == len(order_feats)

      reordered_prods = ops.loc[
          (ops['reordered']==1), 
          'product_id'
          ].unique()
      for prodid in reordered_prods:
        prodmatches = ops[ops['product_id']==prodid]
        # Each user, reordered_prod pair has a unique id, and 
        # corresponds to norders - 1 vector rows 
        seqid = next_seqid
        next_seqid += 1
        for focal_order_ix in range(1, norders):
          # oh god, the inner loops...
          focal_order = group.iloc[focal_order_ix]
          reordered = (prodmatches['order_id']==focal_order.order_id).any()
          # user id, order_number, eval set, etc. in order_feats (for now)
          row = [next_seqid, prodid,
              reordered,
          ]
          row += order_feats[focal_order_ix-1]
          row += ops_feats[focal_order_ix-1]
          rows.append(row)
      users += 1
      if limit is not None and users >= limit:
        logging.info('Stopping vectorization early due to limit reached')
        break
      if next_seqid >= nextlog:
        t1 = time.time()
        logging.info('Finished vectorizing seq {}. Users processed={}. Elapsed={:.1f}s'.format(
          next_seqid, users, t1-t0))
        t0 = time.time()
        nextlog += log_every


    return rows



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--truncate', type=int, default=None,
      help='How much to truncate db on initial load (default: None)')
  parser.add_argument('--lim', type=int, default=None,
      help='Limit the number of users to vectorize. Default none.')
  parser.add_argument('-o', '--output-path', default='vectors.pickle')
  # XXX: not implemented
  parser.add_argument('--user-offset', default=0, 
    help='Skip this many user ids before vectorizing')
  args = parser.parse_args()
  logging.info('Loading db')
  db = BasketDB.load(truncate=args.truncate)
  vz = OrderSequenceVectorizer(db)
  logging.info('Vectorizing')
  t0 = time.time()
  df = vz.get_vector_df(limit=args.lim)
  t1 = time.time()
  logging.info('Finished with {} vector rows'.format(len(df)))
  logging.info('Vectorization took {:.1f}s'.format(t1-t0))
  df.to_pickle(args.output_path)

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)
  main()
  #cProfile.run('main()', 'profile.txt', sort='tottime')
