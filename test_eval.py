from __future__ import division

import numpy as np
import tensorflow as tf
import pytest

import fscore as fsc

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
    ([0, 0], 0, 0),
    ([0, 0], .1, 1),
]

_montecarlo_nondeterministic_examples = [
    # probs,            thresh, expected_fscore range
    ([.9, .9, .9],      0,      (.9, 1.0)  ), # (these ranges are pretty much eyeballed, so idk)
    ([.9, .9, .9],      .91,    (0, .2)  ),
    ([.01, .9, .9],      .8,    (.9, 1.0)  ),
    ([.01, .9, .9],      .001,    (.6, .9)  ),
]

def test_expected_fscore_montecarlo():
  for (probs, thresh, ef) in _montecarlo_deterministic_examples:
    assert fsc.expected_fscore_montecarlo(probs, thresh, 1) == ef

  for (probs, thresh, (lb, ub)) in _montecarlo_nondeterministic_examples:
    ef = fsc.expected_fscore_montecarlo(probs, thresh, 50)
    assert lb <= ef <= ub

