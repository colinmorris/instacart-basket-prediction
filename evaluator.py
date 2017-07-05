from __future__ import division

import numpy as np
import logging

import constants
from results import OrderResults, Results
from predictor import BasePredictor

class Evaluator(object):

  # TODO: at some point in the chain, may want to be cognizant about
  # eval_set = train/prior/test
  def __init__(self, users):
    self.users = users

  def evaluate(self, predictors, limit=None):
    """predictors is map from name to Predictor object"""
    results = {pname: Results() for pname in predictors.keys()}
    i = 0
    for user in self.users:
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
    return results

