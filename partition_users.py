import tensorflow as tf
import argparse
import random

def random_traintest_split(records, test_frac=.05):
  train = tf.python_io.TFRecordWriter('train.tfrecords')
  test = tf.python_io.TFRecordWriter('test.tfrecords')
  for record in records:
    if random.random() < test_frac:
      test.write(record)
    else:
      train.write(record)
  train.close()
  test.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--record-file', default='users.tfrecords')
  args = parser.parse_args()

  record_iterator = tf.python_io.tf_record_iterator(args.record_file)

  random_traintest_split(record_iterator)

main()
