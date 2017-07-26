#!/usr/bin/env python

from __future__ import division
import xgboost as xgb
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from baskets import common

from dataset import Dataset
import hypers

parser = argparse.ArgumentParser()
parser.add_argument('tag')
args = parser.parse_args()
path = common.resolve_xgboostmodel_path(args.tag)
model = xgb.Booster(model_file=path)
hps = hypers.hps_for_tag(args.tag)

def patch_featnames():
  # this is some riggery and tomfoolery
  model.feature_names = Dataset.feature_names_for_hps(hps)
  model.feature_types = None

patch_featnames()

def plot_importance(importance_type='weight'):
  xgb.plot_importance(model, importance_type=importance_type)


def plot_trees(n=2, ax=None, rankdir='UT'): # 'LR' for left-right
  xgb.plot_tree(model, num_trees=n, rankdir=rankdir, ax=ax)

dpi = 96 * 20
for i in range(20):
  plot_trees(i)
  plt.savefig('trees/tree_{}.png'.format(i), dpi=dpi, bbox_inches='tight')
