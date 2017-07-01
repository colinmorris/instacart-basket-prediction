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
  ts = user.training_sequence_for_pid(HHID, maxlen)
  assert ts['pindex'] == HHID-1 # Translated from 1-indexing to 0-indexing
  # Training sequence starts from second order
  seqlen = ts['seqlen']
  assert seqlen == user.norders - 1
  hh_orders = np.array([0, 3, 4, 6])
  true_label_indices = np.array([3, 4, 6]) - 1
  expected_labels = np.zeros(maxlen)
  expected_labels[true_label_indices] = 1
  assert (ts['labels'] == expected_labels).all()

  lossmask = ts['lossmask']
  assert (lossmask[seqlen:] == 0).all()
  assert (lossmask[:seqlen] == 1).all()
  # TODO: test features


