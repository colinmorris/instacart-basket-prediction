import tensorflow as tf

context_fields = [
    'pid', 'aisleid', 'deptid', 'uid', 'weight',
]
raw_feats = ['previously_ordered', 'days_since_prior', 'dow', 'hour',
      'n_prev_products', 'n_prev_repeats', 'n_prev_reorders']
sequence_fields = ['lossmask', 'labels', ] + raw_feats

def parser(record):
  # return some dict or tuple or something
  pass

ds = tf.contrib.data.TFRecordDataset(fname)
ds = ds.map(parser)

# Wrapper around tf.contrib.Dataset
class BasketDataset(object):
  def __init__(self, fname, hps):
    dataset = tf.contrib.data.TFRecordDataset([fname])
    # Parse me
    dataset = dataset.map(self.parse_record_fn())
    # compute my derived/transformed features

  def parse_record_fn(self):
    def _parse(proto):
      ctx, seq = tf.parse_single_sequence_example(proto, **self.record_spec())
      ctx.update(seq)
      return ctx
    return _parse

  def record_spec(self):
    context_feats = {featname: tf.FixedLenFeature([], 
        (tf.float32 if featname == 'weight' else tf.int64))
      for featname in context_fields}
    seq_feats = {featname: tf.FixedLenSequenceFeature([], tf.int64)
        for featname in sequence_fields}
    return dict(context_features=context_feats, sequence_features=seq_feats)
