from __future__ import division
import numpy as np

def expected_fscore_montecarlo(probs, thresh, ntrials):
  # TODO: Should none handling be incorporated at this level?
  probs = np.asarray(probs)

  p_none = np.product(1-probs)
  predict_none = p_none >= thresh

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
    
