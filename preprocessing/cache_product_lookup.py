import csv
import pickle
import os
import numpy as np

from baskets import common

fname = common.csv_path('products')
lookup = {}
parr = np.empty([49688, 2], dtype=np.int32)
with open(fname) as fcsv:
  reader = csv.reader(fcsv)
  reader.next() # Skip header
  for row in reader:
    try:
      pid, _name, aid, deptid = row
    except ValueError:
      print row
      raise
    pid, aid, deptid = map(int, [pid, aid, deptid])
    parr[pid-1] = [aid, deptid]
    lookup[pid-1] = (aid, deptid)

outname = 'product_lookup.pickle'
outpath = os.path.join(common.DATA_DIR, outname)
with open(outpath, 'w') as f:
  pickle.dump(lookup, f)
arroutname = 'product_lookup.npy'
arroutpath = os.path.join(common.DATA_DIR, arroutname)
np.save(arroutpath, parr)

print 'Wrote lookups with {} products to {}, {}'.format(len(lookup), outpath, arroutpath)

