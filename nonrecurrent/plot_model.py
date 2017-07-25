#!/usr/bin/env python

import xgboost as xgb
import argparse
from matplotlib import pyplot as plt

from baskets import common

parser = argparse.ArgumentParser()
parser.add_argument('tag')
args = parser.parse_args()
path = common.resolve_xgboostmodel_path(args.tag)
model = xgb.Booster(model_file=path)

def plot_importance():
  fig, ax = plt.subplots()
  xgb.plot_importance(model, ax)


def plot_trees(n=2):
  xgb.plot_tree(model, num_trees=n)
