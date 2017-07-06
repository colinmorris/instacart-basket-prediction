from __future__ import division

import numpy as np
import logging
import pickle

import constants
from results import OrderResults, Results
from predictor import BasePredictor

class Evaluator(object):

  # TODO: at some point in the chain, may want to be cognizant about
  # eval_set = train/prior/test
  def __init__(self, users):
    self.users = users

  def evaluate(self, predictors, limit=None, save=False):
    """predictors is map from name to Predictor object"""
    assert not save or len(predictors) == 1, "if save, just use one predictor"
    results = {pname: Results() for pname in predictors.keys()}
    i = 0
    uids = []
    for user in self.users:
      uids.append(user.uid)
      actual = user.last_order_predictable_prods()
      for pname, predictor in predictors.iteritems():
        predicted = predictor.predict_last_order(user)
        if not predicted:
          # spammy warning
          if 0:
            logging.warning('''Predictor {} returned an empty prediction. Helping 
it out by turning it into {{NONE_PRODUCTID}}'''.format(pname))
          predicted = set([constants.NONE_PRODUCTID])
        results[pname].add_result(predicted, actual)
      i += 1
      if limit and i >= limit:
        break

    if save:
      pname, p = predictors.items()[0]
      hist = {}
      for (uid, event) in zip(uids, p.history):
        hist[uid] = event
      fname = 'pdicts/{}_predictions.pickle'.format(pname)
      with open(fname, 'w') as f:
        pickle.dump(hist, f)
      logging.info('Wrote prediction debug info to {}'.format(fname))

    return results

