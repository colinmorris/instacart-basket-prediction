import numpy as np

class NonePredictor(object):

  def should_predict_none(pid_to_prob):
    raise NotImplemented

class PNoneThresholdPredictor(NonePredictor):
  """Decides whether to predict none by calculating P(none) as
      product( (1-P(prod)) for each prod)
    and thresholding it."""
  def __init__(self, thresh):
    self.thresh = thresh

  def should_predict_none(self, pid_to_prob):
    probs = np.array(pid_to_prob.values())
    hatprobs = 1 - probs
    pnone = np.product(hatprobs)
    return pnone >= self.thresh
  def should(self, ptp):
    return self.should_predict_none(ptp)
