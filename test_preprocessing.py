import pytest
import numpy as np

from insta_pb2 import User
from batch_helpers import UserWrapper

TEST_UID = 2455
userinfo = dict(
    norders= 7,
    nprodorders=31,
    # Lots of milk.
    # Recorded in approximately chrono order
    prods=[15598, 27845, 35951, 15429, 27086, 19348,
      33334,
      14233,
      47114,
      27243, 22950, 30066, 32423, 49424,
      # NB: These last two are unique to the final order
      47630, 26209],
)
HHID = 27086 # pid for half-and-half

@pytest.fixture(scope='module')
def user_pb():
  u = User()
  with open('testuser.pb') as f:
    u.ParseFromString(f.read())
  return u

@pytest.fixture()
def user(user_pb):
  return UserWrapper(user_pb)


def test_userwrapper_props(user):
  assert user.norders == userinfo['norders']
  prods = set(userinfo['prods'])
  trainable_prods = prods.difference([47630, 26209])
  theirprods = user.all_pids
  assert len(theirprods) == len(trainable_prods)
  assert theirprods == trainable_prods

def test_half_and_half_trainseq(user):
  maxlen = 100
  x, labels, seqlen, lossmask = user.training_sequence_for_pid(HHID, maxlen)
  # Training sequence starts from second order
  assert seqlen == user.norders - 1
  hh_orders = np.array([0, 3, 4, 6])
  true_label_indices = np.array([3, 4, 6]) - 1
  expected_labels = np.zeros(maxlen)
  expected_labels[true_label_indices] = 1
  assert (labels == expected_labels).all()
  # TODO: test features




'''
  # TODO: for a test like this, maybe makes sense to test the 
  # ndarray forms at the same time?
  def test_half_and_half(self):
    """This user ordered half and half a few times."""
    train = self.train_df
    hhid = 27086 # product_id of half and half
    hh_vecs = train[train['product_id'] == hhid]
    
    # We should have exactly one vector/instance for .5+.5 for
    # each of this user's 7 orders.
    self.assertEqual(len(hh_vecs), self.userinfo['norders'])

    # It was ordered 4 times (including 1st and last order)
    self.assertEqual(hh_vecs['ordered'].sum(), len(hh_orders))
'''
