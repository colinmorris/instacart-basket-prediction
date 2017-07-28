import xgboost as xgb
import numpy as np
import os
import sys

try:
  path = sys.argv[1]
except IndexError:
  path = 'cache/train_5__Wnone.buffer'

dm = xgb.DMatrix(path)
