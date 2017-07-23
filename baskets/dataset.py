import tensorflow as tf

from baskets.hypers import Mode
from baskets import common, data_fields
from baskets.feature_spec import FeatureSpec

# TODO: some big centralized master record of these fields with all the tabular
# fields you could want (shape, dtype, which modes it's used in, context vs. sequence,
# etc.)


INPUT_KEYS = {'pid', 'aisleid', 'deptid', 'features', 'lossmask', 'labels', 'seqlen',
    'uid', 'weight',
    }

context_fields = [
    'pid', 'aisleid', 'deptid', 'uid', 'weight',
]
raw_feats = ['previously_ordered', 'days_since_prior', 'dow', 'hour',
      'n_prev_products', 'n_prev_repeats', 'n_prev_reorders']
sequence_fields = ['lossmask', 'labels', ] + raw_feats

# XXX: Structure of this code is a real mess. Waiting for the temperature to
# fall before commiting to architecture.

class DatasetWrapper(object):
  FIELDS = []
  @classmethod
  def dictify(kls, tensor_seq):
    return {name: tensor for name, tensor in zip(kls.FIELDS, tensor_seq)}

class RawDataset(DatasetWrapper):
  FIELDS = context_fields + sequence_fields
  _mode_to_default_recordname = {
      Mode.training: 'train.tfrecords',
      Mode.inference: 'test.tfrecords',
      Mode.eval: 'validation.tfrecords',}

  def __init__(self, hps, fname=None):
    self.hps = hps
    if fname is None:
      fname = self._mode_to_default_recordname[hps.mode]
    path = common.resolve_vector_recordpath(fname)
    dataset = tf.contrib.data.TFRecordDataset([path], 
        compression_type=common.VECTOR_COMPRESSION_NAME)
    dataset = dataset.map(self.parse_record_fn())
    self.dataset = dataset

  def parse_record_fn(self):
    def _parse(proto):
      spec = self.record_spec()
      ctx, seq = tf.parse_single_sequence_example(proto, **spec)
      ctx.update(seq)
      result = []
      for field in self.FIELDS:
        result.append(ctx[field])
      return result
    return _parse

  def record_spec(self):
    field_dtype = lambda fname: tf.float32 if data_fields.FIELD_LOOKUP[fname].dtype==float \
        else tf.int64
    context_feats = {featname: tf.FixedLenFeature([], field_dtype(featname))
      for featname in context_fields}
    seq_feats = {featname: tf.FixedLenSequenceFeature([], field_dtype(featname))
        for featname in sequence_fields}
    return dict(context_features=context_feats, sequence_features=seq_feats)


class TransformedDataset(DatasetWrapper):
  # keys_for_mode blah blah
  FIELDS = sorted(list(INPUT_KEYS))
  def __init__(self, hps, raw):
    feat_spec = FeatureSpec.for_hps(hps)
    self.feat_spec = feat_spec
    self.dataset = self.featurize(raw, feat_spec)
  
  def featurize(self, raw, feat_spec):
    def _featurize(*ds):
      raw_as_dict = RawDataset.dictify(ds)
      res = []
      for fieldname in self.FIELDS:
        if fieldname == 'features':
          tensor = feat_spec.features_tensor_for_dataset(raw_as_dict)
        elif fieldname == 'seqlen':
          tensor = tf.shape(raw_as_dict['labels'])[0]
        else:
          tensor = raw_as_dict[fieldname]
          if fieldname == 'labels':
            tensor = tf.cast(tensor, tf.float32)
        res.append(tensor)
      return res

    """XXX: It is often convenient to give names to each component of an element, for example if they represent different features of a training example. In addition to tuples, you can use collections.namedtuple or a dictionary mapping strings to tensors to represent a single element of a Dataset."""
    # tf, why you always fuckin lying?
    # Oh, it's only in master. Fine.
    return raw.dataset.map(_featurize)


# TODO: skipping to random offset on resume
# Wrapper around tf.contrib.Dataset
class BasketDataset(DatasetWrapper):
  FIELDS = TransformedDataset.FIELDS
  def __init__(self, hps, fname=None):
    with tf.device('/cpu:0'):
      self.hps = hps
      raw = RawDataset(hps, fname)
      transformed = TransformedDataset(hps, raw)
      # TODO: Feature transformations before or after batching? Does it matter?
      # TODO: normalize. Kinda hairy.
      # batch
      padded_shapes = []
      for field in transformed.FIELDS:
        if field in context_fields or field == 'seqlen':
          shape = []
        elif field in sequence_fields:
          shape = [-1]
        elif field == 'features':
          shape = transformed.feat_spec.shape
        else:
          assert False, "Don't know what to do with field {}".format(field)
        padded_shapes.append(shape)
      self.dataset = transformed.dataset
      self.feature_spec = transformed.feat_spec
      # TODO: does the order of calls to batch/repeat matter? in docs they do repeat first
      if hps.mode == Mode.training:
        # NB: VERY IMPORTANT to shufle before batching. Glad I caught that.
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.padded_batch(hps.batch_size, padded_shapes)
        self.dataset = self.dataset.repeat()
        self.iterator = self.dataset.make_one_shot_iterator()
      else:
        self.dataset = self.dataset.padded_batch(hps.batch_size, padded_shapes)
        self.dataset = self.dataset.repeat(1)
        self.iterator = self.dataset.make_initializable_iterator()
      self.next_element = self.iterator.get_next() 

  def new_epoch_op(self):
    assert self.hps.mode != Mode.training
    return self.iterator.initializer

  @classmethod
  def keys_for_mode(kls, mode):
    # TODO: Being lazy. In future, might be worth dropping data not used in 
    # certain modes (e.g. lossmask for inference)
    all_keys = context_fields + ['lossmask', 'labels', 'features']
    return set(all_keys)

  def model_input_dict(self):
    return self.dictify(self.next_element)

  def __getitem__(self, varname):
    return self.model_input_dict()[varname]



