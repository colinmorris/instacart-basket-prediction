import pytest

from baskets.test_helpers import user, user_pb, TESTUSER_ID, HALF_AND_HALF_ID

import vectorize

def test_make_examples(user):
  egs = list(vectorize.make_examples(user, deepcopy=True))
  
  test_egs = [eg for eg in egs if eg.test]
  assert len(test_egs) == len(user.all_pids)

  eg = test_egs[0]
  assert eg.gfs['hour'] == 15
  assert eg.gfs['days_since_prior'] == 1
  assert eg.gfs['uid'] == TESTUSER_ID

  # "Cereal Honey Bears" is an item ordered only once, in this user's penultimate order.
  HB_PID = 49424
  honeybear_egs = [eg for eg in egs if eg.pid == HB_PID]
  # Honeybears are only eligible for prediction in the final order.
  assert len(honeybear_egs) == 1
  assert honeybear_egs[0].label == 0

  # Half and half was in this user's first basket, so it's a candidate for prediction
  # for all subsequent orders
  hah_egs = [eg for eg in egs if eg.pid == HALF_AND_HALF_ID]
  assert len(hah_egs) == 6
  # Half and half is in 4 of their orders, but one is the first order, which doesn't count
  assert sum(eg.label for eg in hah_egs) == 3

  hah_tests = [eg for eg in hah_egs if eg.test]
  assert len(hah_tests) == 1
