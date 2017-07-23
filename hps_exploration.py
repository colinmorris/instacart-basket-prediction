#!/usr/bin/env python
from __future__ import division
import random
import argparse

from baskets import hypers

hp_exploration_cands = dict(

    rnn_size = {256: .85, 512: .15},
    batch_size = [128, 256, 512],
    learning_rate = [.1, .01, .001, .0001, .00001],
    decay_rate=[.995, .999, .9995, .9999],
    min_learning_rate_denom=[10, 100, 1000],
    product_embedding_size=[0, 8, 32, 64],
    dept_embedding_size=[0, 4],
    aisle_embedding_size=[0, 4, 8, 16],
    l2_weight=[0.00001, 0.000001, 0.0000001, 0],
    recurrent_dropout_prob={.6: .1, .9: .8, .95: .1},
    grad_clip={0.0: .75, 1.0: .25},
    cell={'lstm': .8, 'layer_norm': .15, 'hyper': .05},
    optimizer=['Adam', 'LazyAdam'],
    normalize_features={True: .75, False: .25},
)

ordered_hp_aliases = [
    ('learning_rate', 'lr'),
    ('decay_rate', 'lrd'),
    ('grad_clip', 'gc'),
    ('cell', 'cell'),
    ('rnn_size', 'rnnsz'),
    ('batch_size', 'bs'),
    ('product_embedding_size', 'pemb'),
]

def _sample_dict(hp_dict):
  acc = 0
  x = random.random()
  for (value, prob) in hp_dict.iteritems():
    acc += prob
    if x < acc:
      return value

  assert False


def sample_hps():
  id = '_' + ''.join(chr(random.randint(ord('a'), ord('z'))) for _ in range(4))
  hps = hypers.get_default_hparams()
  for param, cands in hp_exploration_cands.iteritems():
    if param == 'min_learning_rate_denom':
      continue
    if isinstance(cands, list):
      value = random.choice(cands)
    elif isinstance(cands, dict):
      value = _sample_dict(cands)
    else:
      assert False
    setattr(hps, param, value)

  minlr_denom = random.choice(hp_exploration_cands['min_learning_rate_denom'])
  hps.min_learning_rate = hps.learning_rate/minlr_denom

  tag = id
  for (hp, alias) in ordered_hp_aliases:
    val = getattr(hps, hp)
    cands = hp_exploration_cands[hp]
    if isinstance(cands, dict):
      cands = sorted(cands.keys())
    assert val in cands, "Couldn't find value {} in cands for param {}".format(val, hp)
    val_idx = cands.index(val)
    tag += '_{}{}'.format(alias, val_idx)

  return hps, tag

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', type=int, default=10)
  args = parser.parse_args()

  for _ in range(args.n):
    hps, tag = sample_hps()
    hps.num_steps = 12000
    hypers.save_hps(tag, hps)
    print tag

if __name__ == '__main__':
  main()
