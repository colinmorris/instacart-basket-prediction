#!/usr/bin/env python
import tensorflow as tf
import argparse
import random
from collections import defaultdict

from baskets.insta_pb2 import User
from baskets import common

SPLIT_CONFIG = [
    ('train', .95),
    ('test', .01),
    ('validation', .04),
]

def random_traintest_split(records):
  out_dir = common.USER_PB_DIR
  writers = {
      foldname: tf.python_io.TFRecordWriter('{}/{}.tfrecords'.format(out_dir, foldname))
      for (foldname, _frac) in SPLIT_CONFIG
      }
  per_fold = defaultdict(int)
  for record in records:
    x = random.random()
    acc = 0
    for (fold, frac) in SPLIT_CONFIG:
      acc += frac
      if x < acc:
        writers[fold].write(record)
        per_fold[fold] += 1
        break
    else:
      assert False, "this should be unreachable (acc={})".format(acc)
  print "Wrote records with dist: {}".format(per_fold)
  for writer in writers.values():
    writer.close()

def save_testusers(records):
  """Save the kaggle-defined test set users to their own tfrecords file."""
  out = tf.python_io.TFRecordWriter('ktest.tfrecords')
  n = 0
  for record in records:
    user = User()
    user.ParseFromString(record)
    if user.test:
      out.write(record)
      n += 1
  print "Wrote {} records to ktest.tfrecords".format(n)
  out.close()



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--record-file', default='users.tfrecords')
  parser.add_argument('--traintest', action='store_true')
  parser.add_argument('--ktest', action='store_true')
  args = parser.parse_args()
  random.seed(1337)

  recordpath = common.resolve_recordpath(args.record_file)
  record_iterator = tf.python_io.tf_record_iterator(recordpath)

  if args.traintest:
    random_traintest_split(record_iterator)
  elif args.ktest:
    save_testusers(record_iterator)
  else:
    assert False, "nothing to do here"

main()
