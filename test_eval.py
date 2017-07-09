from __future__ import division

import numpy as np
import tensorflow as tf
import pytest

import fscore as fsc
import clever_fscore as cfsc

@pytest.fixture()
def sess():
  return tf.InteractiveSession()

_fscore_examples = [
    # predicted, actual, fscore
    ([0],       [1],    0),
    ([1,1,1],   [1,1,0], 4/5),
    ([0],       [0],    1), # degenerate case
    ([1,1,0],   [1,0,1], .5),
]

def test_fscore():
  for (pred, actual, fscore) in _fscore_examples:
    assert fsc.fscore(np.array(pred), np.array(actual)) == fscore

_montecarlo_deterministic_examples = [
    # probs, thresh, expected_fscore
    ([1, 1], 0, 1),
    ([1, 1], .5, 1),
    ([0, 0], .9, 1),
]

_montecarlo_nondeterministic_examples = [
    # probs,            thresh, expected_fscore range
    ([.9, .9, .9],      .05,      (.85, 1.0)  ), # (these ranges are pretty much eyeballed, so idk)
    ([.9, .9, .9],      .91,    (0, .2)  ),
    ([.01, .9, .9],      .8,    (.85, 1.0)  ),
    ([.01, .9, .9],      .001,    (.55, .9)  ),
]

def test_expected_fscore_montecarlo():
  for (probs, thresh, ef) in _montecarlo_deterministic_examples:
    assert fsc.expected_fscore_montecarlo(probs, thresh, 1) == ef

  for (probs, thresh, (lb, ub)) in _montecarlo_nondeterministic_examples:
    ef = fsc.expected_fscore_montecarlo(probs, thresh, 50)
    assert lb <= ef <= ub

def test_gt_prob():
  gtp = fsc.gt_prob
  probs = np.array([.8, .5])
  assert gtp(np.array([0,0]), probs) == pytest.approx(.1)
  assert gtp(np.array([1,1]), probs) == pytest.approx(.4)
  assert gtp(np.array([0,1]), probs) == pytest.approx(.1)
  assert gtp(np.array([1,0]), probs) == pytest.approx(.4)

def test_exact_fscore_naive():
  probs = [.25]
  ef = fsc.exact_expected_fscore_naive
  assert ef(probs, 0) == 2/3
  assert ef(probs, .26) == 3/4

def test_exact_fscore_clever():
  probs = [.25]
  ef = cfsc.efscore
  assert ef(probs, 0) == 2/3
  assert ef(probs, .26) == 3/4

def test_exact_fscore_agreement():
  """Test whether the naive and 'clever' implementations of exact expected fscore
  agree on a bunch of randomly generated examples of different sizes.
  """
  for nprobs in range(1, 9):
    probs = np.random.rand(nprobs)
    thresh = np.random.rand()
    naive = fsc.exact_expected_fscore_naive(probs, thresh)
    clever = cfsc.efscore(probs, thresh)
    assert naive == pytest.approx(clever), 'Disagreement on probs {} with thresh {}'.format(probs, thresh)

