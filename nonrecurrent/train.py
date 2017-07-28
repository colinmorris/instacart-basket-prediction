#!/usr/bin/env python

from __future__ import division
import argparse
import xgboost as xgb
from collections import defaultdict
import logging
import pickle
import os
import time

from baskets import common
from baskets.time_me import time_me
from baskets.hypers import Mode

from dataset import Dataset
import hypers


P_THRESH = .2
THRESH = (P_THRESH)

counter = 0
def train(traindat, tag, hps):
  valdat = Dataset('validation', hps, mode=Mode.eval)
  # TODO: try set_base_margin (https://github.com/dmlc/xgboost/blob/master/demo/guide-python/boost_from_prediction.py)
  with time_me('Made training dmatrix', mode='stderr'):
    dtrain = traindat.as_dmatrix()
  def quick_fscore(preds, _notused_dtrain):
    global counter
    counter += 1
    if 0 and counter % 5 != 0:
      return 'fscore', 0.0
    with time_me('calculated validation fscore', mode='print'):
      user_counts = defaultdict(lambda : dict(tpos=0, fpos=0, fneg=0))
      uids = valdat.uids
      labels = dval.get_label()
      for i, prob in enumerate(preds):
        uid = uids[i]
        pred = prob >= THRESH
        label = labels[i]
        if pred and label:
          user_counts[uid]['tpos'] += 1
        elif pred and not label:
          user_counts[uid]['fpos'] += 1
        elif label and not pred:
          user_counts[uid]['fneg'] += 1
      fscore_sum = 0
      for uid, res in user_counts.iteritems():
        numerator = 2 * res['tpos']
        denom = numerator + res['fpos'] + res['fneg']
        if denom == 0:
          fscore = 1
        else:
          fscore = numerator / denom
        fscore_sum += fscore
      return 'fscore', fscore_sum / len(user_counts)
    
  dval = valdat.as_dmatrix()
  # If you pass in more than one value to evals, early stopping uses the
  # last one. Because why not.
  watchlist = [(dtrain, 'train'), (dval, 'validation'),]
  #watchlist = [(dval, 'validation'),]

  xgb_params = hypers.xgb_params_from_hps(hps)
  evals_result = {}
  t0 = time.time()
  model = xgb.train(xgb_params, dtrain, hps.rounds, evals=watchlist, 
      early_stopping_rounds=hps.early_stopping_rounds, evals_result=evals_result) #, feval=quick_fscore, maximize=True)

  t1 = time.time()
  model_path = common.resolve_xgboostmodel_path(tag)
  model.save_model(model_path)
  preds = model.predict(dval)
  _, fscore = quick_fscore(preds, None)
  logging.info('Final validation (quick) fscore = {}'.format(fscore))
  resultsdict = dict(fscore=fscore, evals=evals_result, duration=t1-t0)
  res_path = os.path.join(common.XGBOOST_DIR, 'results', tag+'.pickle')
  with open(res_path, 'w') as f:
    pickle.dump(resultsdict, f)

def validate_hps(hps):
  assert hps.max_depth > 0 or hps.grow_policy != 'depthwise'
  assert hps.max_depth > 0 or hps.max_leaves > 0
  if hps.grow_policy == 'lossguide':
    assert hps.tree_method == 'hist'

  if hps.embedding_tag is not None:
    # Nvm, you can have both if you want.
    #assert 'pid' not in hps.onehot_vars
    assert hps.embedding_dimension > 0
  # TODO: add more as they come up

def main():
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('tag')
  parser.add_argument('--train-recordfile', default='train', 
      help='identifier for file with the users to train on (default: train)')
  parser.add_argument('-n', '--n-rounds', type=int, default=50,
      help='Number of rounds of boosting')
  parser.add_argument('--weight', action='store_true')
  args = parser.parse_args()

  try:
    hps = hypers.hps_for_tag(args.tag)
  except hypers.NoHpsDefinedException:
    logging.warn('No hps found for tag {}. Creating and saving some.'.format(args.tag))
    hps = hypers.get_default_hparams()
    hps.train_file = args.train_recordfile
    hps.rounds = args.n_rounds
    hps.weight = args.weight
    hypers.save_hps(args.tag, hps)
  validate_hps(hps)
  dataset = Dataset(hps.train_file, hps)
  with time_me(mode='stderr'):
    train(dataset, args.tag, hps)

if __name__ == '__main__':
  main()
