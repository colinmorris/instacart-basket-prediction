import sys, os.path
import numpy as np
import tensorflow as tf
from baskets.insta_pb2 import User, Order
from baskets import common

rfile = sys.argv[1]
if 'vector' in rfile:
  options = tf.python_io.TFRecordOptions(compression_type=common.VECTOR_COMPRESSION_TYPE)
else:
  options = None
ri = tf.python_io.tf_record_iterator(rfile, options=options)

s = next(ri)

def count():
  global ri
  n = 1
  for _ in ri:
    n += 1
  ri = tf.python_io.tf_record_iterator(rfile, options=options)
  return n

print "Total records = {}".format(count())
