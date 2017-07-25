#!/usr/bin/env python

import argparse
import xgboost as xgb

from baskets import common
from baskets.time_me import time_me

from dataset import Dataset

# Copied from https://www.kaggle.com/nickycan/lb-0-3805009-python-edition
# TODO: Loas params from config files.
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

def train(traindat, params, tag, nrounds):
  valdat = Dataset('validation')
  # TODO: try weight parameter
  # TODO: try set_base_margin (https://github.com/dmlc/xgboost/blob/master/demo/guide-python/boost_from_prediction.py)
  # TODO: early_stopping_rounds
  dtrain = traindat.as_dmatrix()
  dval = valdat.as_dmatrix()
  # If you pass in more than one value to evals, early stopping uses the
  # last one. Because why not.
  watchlist = [(dtrain, 'train'), (dval, 'validation'),]

  model = xgb.train(xgb_params, dtrain, nrounds, evals=watchlist, early_stopping_rounds=10)
  # Set output_margin=True to get logits
  model_path = common.resolve_xgboostmodel_path(tag)
  model.save_model(model_path)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('tag')
  parser.add_argument('--train-recordfile', default='train', 
      help='identifier for file with the users to train on (default: train)')
  parser.add_argument('-n', '--n-rounds', type=int, default=50,
      help='Number of rounds of boosting')
  parser.add_argument('--n-users', type=int, 
      help='Limit number of users tested on (default: no limit)')
  args = parser.parse_args()

  dataset = Dataset(args.train_recordfile, maxlen=args.n_users)
  params = xgb_params
  with time_me(mode='stderr'):
    train(dataset, params, args.tag, args.n_rounds)

if __name__ == '__main__':
  main()
