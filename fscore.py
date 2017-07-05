from __future__ import division
import numpy as np
import itertools
import time

def expected_fscore_montecarlo(probs, thresh, ntrials):
  # TODO: Should none handling be incorporated at this level?
  probs = np.asarray(probs)

  p_none = np.product(1-probs)
  # TODO: should be consistent about whether threshholds are inclusive
  # doesn't really matter in practice, but can mess up tests.
  predict_none = p_none > thresh

  predictions = (probs >= thresh).astype(np.int8)
  fscores = np.zeros(ntrials)
  for i in range(ntrials):
    groundtruth =  (np.random.rand(len(probs)) <= probs)\
      .astype(np.int8)
    fscores[i] = fscore(predictions, groundtruth, predict_none)
  # TODO: look at variance
  return fscores.mean()

def fscore(predicted, actual, none=False):
  tp = (predicted * actual).sum()
  wrong = (predicted != actual).sum()
  # Did we predict 'none' (pid -1)?
  if none:
    # If there were actually no reordered products, none is tp
    if actual.sum() == 0:
      tp += 1
    # Otherwise it's a fp. so to speak.
    else:
      wrong += 1
  if tp == wrong == 0:
    # It doesn't matter if we failed to explicitly predict 'none'. To get here,
    # we must have predicted no products, meaning that 'none' is a freebie.
    return 1
  return (2*tp) / (2*tp + wrong)


def sample_groundtruth(probs):
  return (np.random.rand(len(probs)) <= probs).astype(np.int8)
    
def timeit(f):
  def timed(*args, **kw):
    ts = time.time()
    result = f(*args, **kw)
    te = time.time()
    print 'took: %2.4f sec' % \
      (te-ts)
    return result
  return timed

def exact_expected_fscore_naive(probs, thresh):
  """NB: This algorithm is exponential in the size of probs!
  Based on initial measurements, less than 15 items is
  sub-second. 16 = 2s, 17=4s, 18=8s, and, well, you know
  the rest...
  possible relaxation to allow larger number of products:
  force items with sufficiently low probs (e.g. < 1%) off
  in groundtruths.
  """
  probs = np.asarray(probs)
  n = len(probs)
  expected = 0
  p_none = np.product(1-probs)
  predict_none = p_none > thresh
  predictions = (probs >= thresh).astype(np.int8)
  for gt in itertools.product([0,1], repeat=n):
    gt = np.array(gt)
    fs = fscore(predictions, gt, predict_none)
    p = gt_prob(gt, probs)
    expected += fs * p
  return expected
  
def gt_prob(gt, probs):
  return np.product( np.abs((gt-1) + probs) )

if __name__ == '__main__':
  exact = exact_expected_fscore_naive
  ex = timeit(exact)
  maxitems = 19
  for n in range(2, maxitems):
    print "n = {}".format(n)
    probs = np.random.rand(n)
    ex(probs, .3)
