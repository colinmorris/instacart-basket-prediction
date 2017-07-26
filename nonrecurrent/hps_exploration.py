#!/usr/bin/env python
"""mostly cribbed from script of same name up one level"""
from __future__ import division
import random
import argparse

import hypers

hp_exploration_cands = dict(

    weight = [True, False],
    soft_weights = [True, False],

    eta = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    # (0 = no limit)
    max_depth = [4, 5, 6, 7, 8, 10], # 0 not working
    min_child_weight = [0.5, 1, 2, 5, 10, 20],
    gamma = [0, 0.1, 0.33, 0.5, 0.7, 0.9, 1.2],
    subsample = [0.4, 0.5, 0.66, 0.76, 0.9, 1.0],
    colsample_bytree = [0.3, 0.5, 0.7, 0.85, 0.95, 1.0],
    colsample_bylevel = {0.3:.1, 0.5:.1, 0.85:.1, 1.0:.7},
    reg_lambda = [0.1, 1, 10, 30],
    alpha = [0, 1e-06, 1e-05, 1e-04, 1e-03],
    max_delta_step = {0: .9, 1: .05, 10: .05},
    tree_method = {'approx': 1.0, 'hist': 0.0},
    scale_pos_weight = {1: .9, 2: .05, 10: .05},
    grow_policy = {'depthwise': 1.0, 'lossguide': 0.0},
)

def _sample_dict(hp_dict):
  acc = 0
  x = random.random()
  for (value, prob) in hp_dict.iteritems():
    acc += prob
    if x < acc:
      return value

  assert False

def sample_hps():
  id = 'x_' + ''.join(chr(random.randint(ord('a'), ord('z'))) for _ in range(4))
  hps = hypers.get_default_hparams()
  for param, cands in hp_exploration_cands.iteritems():
    if isinstance(cands, list):
      value = random.choice(cands)
    elif isinstance(cands, dict):
      value = _sample_dict(cands)
    else:
      assert False
    setattr(hps, param, value)

  # Hackety hack
  x =random.random()
  if x < .3:
    hps.onehot_vars = [None, 'pid', 'aisleid', 'deptid']
  elif x < .6:
    hps.onehot_vars = [None, 'aisleid', 'deptid']
  else:
    hps.onehot_vars = [None]

  # XXX: this constellation of options seems not to work right now...
  # more hack
  if hps.max_depth == 0:
    hps.grow_policy = 'lossguide'
  if hps.grow_policy == 'lossguide':
    hps.tree_method = 'hist'

  tag = id
  return hps, tag

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', type=int, default=10)
  args = parser.parse_args()

  for _ in range(args.n):
    hps, tag = sample_hps()
    hps.rounds = 50
    hypers.save_hps(tag, hps)
    print tag

if __name__ == '__main__':
  main()
