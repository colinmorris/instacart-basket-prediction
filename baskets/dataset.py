import tensorflow as tf

context_fields = [
    'pid', 'aisleid', 'deptid', 'uid', 'seqlen', 'weight',
]
raw_feats = ['previously_ordered', 'days_since_prior', 'dow', 'hour',
      'n_prev_products', 'n_prev_repeats', 'n_prev_reorders']
# I guess technically lossmask is computable from labels? But eh.
sequence_fields = ['lossmask', 'labels', ] # + raw_feats

def parser(record):
  # return some dict or tuple or something
  pass

ds = tf.contrib.data.TFRecordDataset(fname)
ds = ds.map(parser)

