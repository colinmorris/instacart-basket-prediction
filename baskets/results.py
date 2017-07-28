from __future__ import division

import pandas as pd

class OrderResults(object):
  """Outcome of prediction for one order"""
  def __init__(self, tp, fp, fn):
    self.tp = tp
    self.fp = fp
    self.fn = fn

  def __repr__(self):
    return str(self)

  def __str__(self):
    return 'tp:{}, fp:{}, fn:{}, precision={:.2f}, recall={:.2f}, fscore={:.2f}'.format(
        self.tp, self.fp, self.fn, self.precision, self.recall, self.fscore)

  @classmethod
  def for_pids(kls, predicted, actual):
    predicted = set(predicted)
    actual = set(actual)
    fp = len(predicted.difference(actual))
    fn = len(actual.difference(predicted))
    tp = len(predicted.intersection(actual))
    return kls(tp, fp, fn)

  # TODO: why is this implemented in 2 places? (cf. fscore.py)
  @property
  def fscore(self):
    prec = self.precision
    rec = self.recall
    num = prec * rec
    denom = prec + rec
    if denom == 0:
      actual_pos = self.tp + self.fn
      pred_pos = self.tp + self.fp
      # Special case: There were no products to predict, and we predicted none.
      if actual_pos == 0 and pred_pos == 0:
        assert False, "Shouldn't be possible for actual_pos to be 0. Use dummy pid."
      else:
        return 0
    return 2 * (num / denom)

  @property
  def precision(self):
    denom = self.tp + self.fp
    return 0 if denom == 0 else (self.tp / denom)
  @property
  def recall(self):
    denom = self.tp + self.fn
    return 0 if denom == 0 else (self.tp / denom)

# TODO: maybe this should be a thin wrapper around a df with cols for prec, rec, fscore
class Results(object):

  def __init__(self, subresults=None):
    self.subs = subresults or []

  def add_result(self, predicted, actual):
    # TODO: it might be worth storing the actual/predicted values, at least
    # in some cases for the purposes of debugging/postmortem
    sub = OrderResults.for_pids(predicted_actual)
    self.subs.append(sub)

  def to_df(self):
    col_attrs = ['fscore', 'precision', 'recall']
    col_data = []
    for attr in col_attrs:
      col = [getattr(sub, attr) for sub in self.subs]
      col_data.append(col)
    df = pd.DataFrame(col_data)
    df = df.T
    df.columns = col_attrs
    return df

  @property
  def fscores(self):
    return [sub.fscore for sub in self.subs]

  def __repr__(self):
    return 'Results over {:,} orders with mean fscore = {:.4f}'.format(
        len(self.fscores),
        self.fscores.mean()
    )
