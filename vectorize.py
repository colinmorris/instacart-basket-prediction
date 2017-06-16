import tensorflow as tf
import pandas as pd
import numpy as np
import logging

from basket_db import BasketDB

class OrderSequenceVectorizer(object):

  def __init__(self, db):
    self.db = db

  def ops_for_orders(self, orders):
    """returns ops df. input is a subset of orders df"""
    return self.db.ops.loc[orders['order_id']]

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
    logging.debug('Example vector list:\n{}'.format(vlists[0]))
    cols = (
        ['seqid', 'prodid', 'reordered',]
        + self.db.orders.columns.tolist()
        + ['prev_prods', 'prev_reorders']
        )
    df = pd.DataFrame(vlists, columns=cols)
    return df
        

  def vector_lists(self, limit=None):
    orders = self.db.orders
    orders_by_user = orders.groupby('user_id')
    # TODO: maybe should be an ndarray? 
    # but would need uniform type?
    # TODO: memory of this might be crazy. May need to build incrementally. Ugh.
    rows = []
    next_seqid = 0
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
      if limit is not None and next_seqid >= limit:
        logging.info('Stopping vectorization early due to limit reached')
        break
    return rows



def main():
  logging.info('Loading db')
  db = BasketDB.load(truncate=1000)
  vz = OrderSequenceVectorizer(db)
  logging.info('Vectorizing')
  df = vz.get_vector_df(limit=None)
  df.to_pickle('vectors.pickle')

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)
  main()
