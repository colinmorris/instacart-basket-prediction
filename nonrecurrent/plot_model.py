#!/usr/bin/env python

import xgboost as xgb
import argparse
from matplotlib import pyplot as plt

from baskets import common

from dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('tag')
args = parser.parse_args()
path = common.resolve_xgboostmodel_path(args.tag)
model = xgb.Booster(model_file=path)

def patch_featnames():
  # this is some riggery and tomfoolery
  model.feature_names = Dataset.feat_cols
  model.feature_types = None

patch_featnames()

def plot_importance(importance_type='weight'):
  xgb.plot_importance(model, importance_type=importance_type)


def plot_trees(n=2):
  xgb.plot_tree(model, num_trees=n)
