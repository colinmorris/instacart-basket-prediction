from __future__ import division

import numpy as np

from results import OrderResults, Results

class Evaluator(object):

  # TODO: at some point in the chain, may want to be cognizant about
  # eval_set = train/prior/test
  def __init__(self, test_data):
    self.dat = test_data

  def evaluate(self, predictor):
    results = Results()
    for (uid, df) in self.dat.groupby('user_id'):
      last_order_num = df['order_number'].max()
      history = df[df['order_number'] < last_order_num]
      predicted = predictor.predict_next_order(history)
      actual = df.loc[
              (df['order_number'] == last_order_num) & (df['reordered'] == 1),
              'prodid'].tolist()
      results.add_result(predicted, actual)
    return results

