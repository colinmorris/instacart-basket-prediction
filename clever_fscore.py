"""
~O(n^3) algorithm for exact expected fscore, as described in Jansche 2007:
  http://clair.eecs.umich.edu/aan/paper.php?paper_id=P07-1093#pdf

(Described elsewhere in the code as O(n^4), because that's the cost of
finding the optimum threshold.)
"""
from __future__ import division
from collections import defaultdict
import numpy as np
import time

def efscore(probs, thresh):
  probs = np.asarray(probs)
  p_none = np.product(1-probs)
  predict_none = p_none > thresh
  predictions = (probs >= thresh).astype(np.int8)

  dfa = FscoreDFA(probs, predictions, predict_none)
  dfa.fill()

  # TODO: be vewy caweful about none handling
  AP = np.sum(predictions) + int(predict_none)# "alleged positives"
  n = len(probs)
  EF = 0
  # Usinng the nomenclature of dumb O^4 paper
  for T in range(n+1): # Number of actual/groundtruth positives.
    for A in range(T+1): # Number of true positives
      # Special case for groundtruth positives = 0. The 'none' label.
      A_ = A
      T_ = T
      if T == 0:
        T = 1
        A = int(predict_none)
      f_denom = (AP + T)
      if f_denom == 0:
        assert False, "unpossible"
        fs = 1
      else:
        fs = 2*A / f_denom
      # fs is the fscore when we have this many true pos/gt pos
      # Now just need to know the probability weight of (A, T)
      ant_prob = dfa[A_, T_]
      EF += ant_prob * fs
  return EF



class FscoreDFA(object):

  ddlambda = lambda _ : 0.0
  def __init__(self, probs, predictions, predicted_none):
    self.probs = probs
    self.predictions = predictions

    self.states = defaultdict(self.ddlambda)
    self.states[0,0] = 1

  def __getitem__(self, statetuple):
    """(A, T)"""
    return self.states[statetuple]

  def fill(self):
    for prob, predicted in zip(self.probs, self.predictions):
      self.percolate(prob, predicted)

  def percolate(self, prob, predicted):
    nexts = defaultdict(self.ddlambda, [(k, 0.0) for k in self.states.keys()])
    p, phat = prob, 1-prob
    for (A, T), stateprob in self.states.iteritems():
      # if this one is true...
      state_pos = (A+predicted), (T+1)
      state_neg = (A, T)

      nexts[state_pos] += stateprob * p
      nexts[state_neg] += stateprob * phat
    self.states = nexts

def timeit(f):
  def timed(*args, **kw):
    ts = time.time()
    result = f(*args, **kw)
    te = time.time()
    print 'took: %2.4f sec' % \
      (te-ts)
    return result
  return timed

# omg this is fast as fuck. Can get past 150
# and stay under .5s. I'm sorry for laughing
# at your O(n^4) algorithm, Jansche.
if __name__ == '__main__':
  exact = efscore
  ex = timeit(exact)
  maxitems = 510
  for n in range(2, maxitems, 50):
    print "n = {}".format(n)
    probs = np.random.rand(n)
    ex(probs, .3)
