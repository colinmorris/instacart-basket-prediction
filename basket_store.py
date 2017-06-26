"""Basically v2 of basket_db.py, backed by HDFStore
"""

import pandas as pd
import numpy as np

import logging

class Store(object):
  """Wrapper around hdfstore with instacart data"""
  PATH = 'store.h5'

  def __init__(self):
    self.store = pd.HDFStore(self.PATH)

  def all_uids(self):
    try:
      return self.store['uids']
    except KeyError:
      logging.warning('No uids series. Creating from scratch.')
    uids = self.store['orders']['user_id'].unique()
    uids = pd.Series(uids)
    self.store['uids'] = uids
    return uids

  def train_df(self, uidlist=None):
    """(uidlist parameter is for testing purposes only)"""
    if uidlist is None:
      uids = self.store['train_uids']
      uidlist = uids.tolist()
    # TODO: may later want to do some kind of batching at this level
    # (see select's chunksize kwarg)
    df = self.store.select(
        'uprods',
        where='user_id = uidlist'
    )
    # shuffle
    df = df.sample(frac=1)
    return df

  def enrich_train_df(self, df):
    # The basic "uprods" table has minimal columns (uid, prodid, orderid, ordered)
    # We need to fill in the rest.
    ordercols = ['order_number', 'order_dow', 'order_hour_of_day', 
        'days_since_prior_order',]
    orders = self.store['orders'] # TODO: select?

    dfcols = ['product_id', 'ordered', 'order_id']
    df = pd.merge(df[dfcols], orders,
        on='order_id', copy=False)
    
    # Counting items in previoud order
    # XXX: this suuuuucks. Giving up for now. Need to return to this.
    return df
    now = df[
        #df['order_number'] > 1,
        ['order_id', 'user_id', 'order_number']
    ]#.drop_duplicates()
    prev = now.copy()
    prev['order_number'] -= 1
    to_prev = pd.merge(prev, orders, how='left', 
        on=['user_id', 'order_number'],
        suffixes=('','_b'))
    #import pdb; pdb.set_trace()
    cols = ['order_id', 'order_id_b']
    to_prev = to_prev[cols]
    noprev = to_prev['order_id_b'].isnull()
    df['prev_prods'] = 0 # TODO: might want a better default value
    df.loc[~noprev, 'prev_prods'] = self.store['prods_per_order'].loc[
        to_prev.loc[~noprev, 'order_id_b']]
    df['prev_reorders'] = 0 # TODO: might want a better default value
    df.loc[~noprev, 'prev_reorders'] = self.store['reorders_per_order'].loc[
        to_prev.loc[~noprev, 'order_id_b']]

    return df



def reload_upo(s):
  store = s.store
  upo = store['upo']
  store.remove('upo')
  # TODO 1: May want more data cols later
  # TODO 2: Should make sure there's a simple way to rebuild whole store from 
  # scratch later. Apparently hdf is bad with reclaiming space/fragmenting.
  store.put('upo', upo, format='table', data_columns=['user_id'])


if __name__ == '__main__':
  s = Store()
  #t = s.train_df()
  #t2 = s.enrich_train_df(t)
