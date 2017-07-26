import sys
import pickle
import os


tag = sys.argv[1]

path = os.path.join('results', tag+'.pickle')
with open(path) as f:
  res = pickle.load(f)
