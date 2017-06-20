from __future__ import division

import pandas as pd

class OrderResults(object):
  """Outcome of prediction for one order"""
  def __init__(self, tp, fp, fn):
    self.tp = tp
    self.fp = fp
    self.fn = fn

  @property
  def fscore(self):
    prec = self.precision
    rec = self.recall
    num = prec * rec
    denom = prec + rec
    if denom == 0:
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

  def __init__(self, subresults=[]):
    self.subs = subresults

  def add_result(self, predicted, actual):
    # TODO: it might be worth storing the actual/predicted values, at least
    # in some cases for the purposes of debugging/postmortem
    predicted = set(predicted)
    actual = set(actual)
    fp = len(predicted.difference(actual))
    fn = len(actual.difference(predicted))
    tp = len(predicted.intersection(actual))
    sub = OrderResults(tp, fp, fn)
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
