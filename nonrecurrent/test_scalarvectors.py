from __future__ import division
import pytest
import numpy as np

from baskets.test_helpers import TESTUSER_ID, HALF_AND_HALF_ID
from baskets import constants

from dataset import Dataset
import hypers
import fields

@pytest.fixture()
def dataset():
  return Dataset('testuser', hypers.get_default_hparams())

@pytest.fixture()
def records(dataset):
  return dataset.records


FIELD_TO_RANGE = dict(
    days_since_prior = (0, 30),
    dow = (0, 6),
    hour = (0, 23),
    n_prev_orders = (1, 99),
    n_prev_focals = (1, 99),
    orders_since_focal = (1, 99),
    orders_since_first_focal = (1, 99),
    pid = (1, 49688),
    aisleid = (1, 135),
    deptid = (1, 22),
    label = (0, 1),
)

def test_field_values_fall_in_allowed_ranges(records):
  for field, (lower, upper) in FIELD_TO_RANGE.iteritems():
    vals = records[field]
    assert (lower <= vals).all()
    assert (vals <= upper).all()

  # All fields should be non-negative
  for field in fields.all_fields:
    vals = records[field]
    assert (vals >= 0).all() or np.isnan(vals).any(), 'Bad values for field {}'.format(field)

# (I ommitted a few generic fields I was too lazy to calculate)
TESTUSER_EXPECTED_GENERIC_FIELDVALUES = dict(
    days_since_prior = 1,
    hour = 15,
    n_prev_repeats = 0,
    uid = TESTUSER_ID,
    orderid = 817343,
    prev_order_size = 5,
    n_prev_orders = 6,

    n_distinct_prods = 14,
    n_singleton_orders = 0,
    order_history_days = 10,
    n_30day_intervals = 0,
)

def test_testuser_generic_fields(records):
  for generic_field, expected_val in TESTUSER_EXPECTED_GENERIC_FIELDVALUES.iteritems():
    exemplar = records[generic_field][0]
    assert exemplar == expected_val, "Mismatch on field {}".format(generic_field)
    # These fields should be uniform across all this user's vectors
    assert (records[generic_field] == exemplar).all()

HALF_AND_HALF_EXPECTED_PRODUCT_FIELDVALUES = dict(
    prev_cartorder = np.nan,
    n_consecutive_prev_focal_orders = 0,
    n_prev_focals = 3,
    orders_since_focal = 2,
    days_since_focal = 2,
    orders_since_first_focal = 6,
    days_since_first_focal = 10,
    n_prev_focals_this_hour = 0,
    avg_focal_order_size = (6 + 3 + 6)/3,
    label = 1,

    n_singleton_focal_orders = 0,
    n_30day_focal_intervals = 0,
    n_30days_since_last_focal = 0,
)

def test_testuser_halfandhalf(records):
  hah_record_indices = np.flatnonzero(records['pid'] == HALF_AND_HALF_ID)
  assert hah_record_indices.shape == (1,)
  hah_record = records[hah_record_indices[0]]
  for field, expected_value in HALF_AND_HALF_EXPECTED_PRODUCT_FIELDVALUES.iteritems():
    if np.isnan(expected_value):
      assert np.isnan(hah_record[field])
    else:
      assert hah_record[field] == expected_value, "Mismatch on field {}".format(field)

  if 1:
    for ff in fields.frecency_feats:
      print '{} = {}'.format(ff, hah_record[ff])

