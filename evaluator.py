from __future__ import division

import numpy as np

from results import OrderResults, Results
from predictor import BasePredictor

class Evaluator(object):

  # TODO: at some point in the chain, may want to be cognizant about
  # eval_set = train/prior/test
  def __init__(self, users):
    self.users = users

  def evaluate(self, predictors, limit=None):
    scalar = isinstance(predictors, BasePredictor)
    if scalar:
      predictors = [predictors]
    results = [Results() for _ in predictors]
    i = 0
    for user in self.users:
      actual = user.last_order_predictable_prods()
      for pred_idx, predictor in enumerate(predictors):
        predicted = predictor.predict_last_order(user)
        results[pred_idx].add_result(predicted, actual)
      i += 1
      if limit and i >= limit:
        break
    return results[0] if scalar else results

