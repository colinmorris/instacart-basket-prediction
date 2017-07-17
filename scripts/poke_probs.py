from __future__ import division
import pickle
import sys
import pandas as pd
import numpy as np

pd.options.display.width = 120

fname = sys.argv[1]

with open(fname) as f:
  tp = pickle.load(f)

print "Loaded dict with {} users".format(len(tp))
nprobs = sum( len(probs) for probs in tp.itervalues() )
print "# product/prob pairs = {} (avg. per user = {:.1}".format(
    nprobs, nprobs/len(tp)
    )

uid_to_nprods = {u: len(probs) for (u, probs) in tp.iteritems()}
s = pd.Series(uid_to_nprods)

def foo():
  cols = ['count', 'min', 'ten', 'quart', 'median', 'threequart', 'ninety', 'max']
  n = len(tp)
  dat = np.zeros( (n, len(cols)) )
  for i, (uid, probmap) in enumerate(tp.iteritems()):
    dat[i,0] = len(probmap)
    probs = np.array(probmap.values())
    dat[i,1] = np.min(probs)
    dat[i,2:7] = np.percentile(probs, [10, 25, 50, 75, 90])
    dat[i,7] = np.max(probs)

  df2 = pd.DataFrame(dat, columns=cols)
  return df2

df = foo()
