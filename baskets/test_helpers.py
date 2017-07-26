import os
import pytest
import tensorflow as tf

from baskets.insta_pb2 import User
from baskets import common
from baskets.user_wrapper import UserWrapper

TESTUSER_ID = 2455 

# The product id of Half and Half. It comes up a lot in testing, for some reason.
HALF_AND_HALF_ID = 27086

@pytest.fixture()
def user_pb():
  u = User()
  testuser_path = os.path.join(common.DATA_DIR, 'testuser.pb')
  with open(testuser_path) as f:
    u.ParseFromString(f.read())
  return u

@pytest.fixture()
def user(user_pb):
  return UserWrapper(user_pb)

@pytest.fixture()
def sess():
  return tf.InteractiveSession()
