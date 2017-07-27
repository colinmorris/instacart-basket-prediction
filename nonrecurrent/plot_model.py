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
  """
  How the importance is calculated: either "weight", "gain", or "cover" 
    "weight" is the number of times a feature appears in a tree 
    "gain" is the average"gain"of splits which use the feature 
    "cover" is the average coverage of splits which use the feature
      where coverage is defined as the number of samples affected by the split
  """
  xgb.plot_importance(model, importance_type=importance_type,
      max_num_features=40,
      )

def plot_trees(n=2, ax=None, rankdir='UT'): # 'LR' for left-right
  xgb.plot_tree(model, num_trees=n, rankdir=rankdir, ax=ax)

def trees(n):
  dpi = 96 * 20
  for i in range(n):
    plot_trees(i)
    plt.savefig('trees/tree_{}.png'.format(i), dpi=dpi, bbox_inches='tight')

if 1:
  for imp in ['weight', 'gain', 'cover']:
    plot_importance(imp)
    plt.savefig('feature_importance_{}_{}.png'.format(imp, args.tag),
        dpi=144,
        )
