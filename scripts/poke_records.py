import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
from insta_pb2 import User, Order

rfile = sys.argv[1]
ri = tf.python_io.tf_record_iterator(rfile)

s = next(ri)
u = User()
u.ParseFromString(s)

def count():
  global ri
  n = 1
  for _ in ri:
    n += 1
  ri = tf.python_io.tf_record_iterator(rfile)
  return n

def get_norders_dist():
  norders = []
  for ustr in ri:
    u = User()
    u.ParseFromString(ustr)
    norders.append( len(u.orders) )
  return np.array(norders)

print "Total users = {}".format(count())

nords = get_norders_dist()
