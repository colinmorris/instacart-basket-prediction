import unittest
import nose2

from basket_store import Store

class TestPreprocessing(unittest.TestCase):

  TEST_UID = 2455
  userinfo = dict(
      norders= 7,
      nprodorders=31,
      # Lots of milk.
      # Recorded in approximately chrono order
      prods=[15598, 27845, 3591, 15429, 27086, 19348,
        33334,
        14233,
        47114,
        27243, 22950, 30066, 32423, 49424,
        47630, 26209],
    )

  @classmethod
  def setUpClass(kls):
    kls.store = Store()
    kls.train_df = kls.store.train_df(uidlist=[kls.TEST_UID])

  def setUp(self):
    pass
  # tearDown

  def test_train_df_topology(self):
    train = self.train_df
    oids = train['order_id'].unique()
    # 7 orders (including last - included in this case because this user
    # belongs to the train rather than test fold)
    self.assertEqual(len(oids), self.userinfo['norders'])

    unique_prods = len(self.userinfo['prods'])
    pids = train['product_id'].unique()
    # Some products may be ordered but not reordered and shouldn't appear?
    # Actually, now that I think of it, no, it shouldn't work like that at all.
    # We should have entries for any product ordered at least once.
    self.assertEqual(len(pids), unique_prods,
        'Train data vectors had {} unique products. Should have been {}'.format(
          len(pids), unique_prods
          )
        )

  # TODO: for a test like this, maybe makes sense to test the 
  # ndarray forms at the same time?
  def test_half_and_half(self):
    """This user ordered half and half a few times."""
    hh_orders = [0, 3, 4, 6]
    train = self.train_df
    hhid = 27086 # product_id of half and half
    hh_vecs = train[train['product_id'] == hhid]
    
    # We should have exactly one vector/instance for .5+.5 for
    # each of this user's 7 orders.
    self.assertEqual(len(hh_vecs), self.userinfo['norders'])

    # It was ordered 4 times (including 1st and last order)
    self.assertEqual(hh_vecs['ordered'].sum(), len(hh_orders))


if __name__ == '__main__':
  nose2.main()
