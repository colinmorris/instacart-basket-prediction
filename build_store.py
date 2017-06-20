import time
import logging
import pandas as pd
import numpy as np

from basket_store import Store

# TODO: make me not mini :(
MINI = 8000

def load_orders():
  orders = pd.read_pickle('orders.pickle')
  orders['eval_set'] = orders['eval_set'].astype('category')
  orders['order_dow'] = orders['order_dow'].astype(np.int8)
  orders['order_hour_of_day'] = orders['order_hour_of_day'].astype(np.int8)
  orders['order_number'] = orders['order_number'].astype(np.int8)
  for smallcol in ['user_id', 'order_id']:
    orders[smallcol] = orders[smallcol].astype(np.int32)
    orders['days_since_prior_order'] = orders['days_since_prior_order'].astype(np.float32)

  if MINI:
    orders = orders[orders['user_id'] <= MINI]
  return orders

def load_ops():
  ops = pd.read_pickle('ops.pickle')
  # (No user-id column, so can't apply MINI here)
  ops = ops.reset_index(drop=True)
  return ops

def load_uprods(store):
  # See "basketdb noodling" ipynb
  userprods = pd.read_pickle('userprods.pickle')
  userprods = userprods.reset_index(drop=True)
  if MINI:
    userprods = userprods[userprods['user_id'] <= MINI]

  # TODO: not sure about this minimal indexing thing, but let's try for now.
  ordercols = ['order_id', 'user_id']
  uprods = pd.merge(userprods, store['orders'][ordercols],
      on='user_id', copy=False)

  opcols = ['order_id', 'product_id']
  ops = store['ops'][opcols]
  up2 = pd.merge(uprods, ops,
      how='left',
      on=['order_id', 'product_id'],
      copy=False,
      indicator=True
  )
  up2['ordered'] = up2['_merge']=='both'
  up2.drop('_merge', axis=1, inplace=True)
  return up2

# Out of 206k total users
TEST_USERS = 1000
TRAIN_USERS = 5000
def augment_uid_data(store):
  uids = load_uids(store)
  ntot = TEST_USERS + TRAIN_USERS
  subset = np.random.choice(uids, ntot, replace=False)

  testers = pd.Series(subset[:TEST_USERS])
  trainers = pd.Series(subset[TEST_USERS:])

  store['test_uids'] = testers
  store['train_uids'] = trainers

def load_uids(store):
  try:
    return store['uids']
  except KeyError:
    logging.warning('No uids series. Creating from scratch.')
  uids = store['orders']['user_id'].unique()
  uids = pd.Series(uids)
  store['uids'] = uids
  return uids

# TODO: not sure if these actually need to be persisted
def load_order_feats(store):
  ops = store['ops']
  store['prods_per_order'] = ops.groupby('order_id').size()
  store['reorders_per_order'] = ops.groupby('order_id')['reordered'].sum()

  # Make mapping from order to previous order
  # XXX: maybe not necessary?
  return
  ordercols = ['order_id', 'order_number', 'user_id']
  orders = store['orders'][ordercols]
  odub = pd.merge(orders, orders, 
      on='user_id', suffixes=('', '_b')
  order_to_prev = odub[
    (odub['order_number'])] # ...


# This is all pretty hacky and relies on some accumulation of previous state.
def main():
  t0 = time.time()
  store = pd.HDFStore(Store.PATH)

  # Eh, removing these checks for now. Can get in a state where the keys are
  # present with a None value because of some earlier botched run.
  #if 'orders' not in store:
  logging.info('Adding orders')
  # Don't need to set order_id as data_column because it's the index?
  store.put('orders', load_orders(), format='table')

  #if 'ops' not in store:
  logging.info('Adding ops')
  store.put('ops', load_ops())

  if 'uprods' not in store:
    logging.info("What's uprods?")
    store.put('uprods', load_uprods(store), data_columns=['user_id'], format='table')

  # TODO: only do this if necessary?
  logging.info('Augmenting uid data')
  augment_uid_data(store)

  logging.info('Loading per-order features')
  load_order_feats(store)

  # TODO: views of train/test data? also make sure train data is shuffled, to avoid
  # correlated batches

  t1 = time.time()
  logging.info('Finished in {:.1f}s'.format(t1-t0))

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main()
