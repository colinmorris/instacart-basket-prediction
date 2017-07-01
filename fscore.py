from __future__ import division
import numpy as np

def expected_fscore_montecarlo(probs, thresh, ntrials):
  # TODO: Should none handling be incorporated at this level?
  probs = np.asarray(probs)
  predictions = (probs >= thresh).astype(np.int8)
  fscores = np.zeros(ntrials)
  for i in range(ntrials):
    groundtruth =  (np.random.rand(len(probs)) <= probs)\
      .astype(np.int8)
    fscores[i] = fscore(predictions, groundtruth)
  # TODO: look at variance
  return fscores.mean()

def fscore(predicted, actual):
  tp = (predicted * actual).sum()
  wrong = (predicted != actual).sum()
  if tp == wrong == 0:
    return 1
  return (2*tp) / (2*tp + wrong)


def sample_groundtruth(probs):
  return (np.random.rand(len(probs)) <= probs).astype(np.int8)
    
