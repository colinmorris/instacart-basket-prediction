import tensorflow as tf
import features
import sequence_example_lib as s_e_l

def fetch_batchdat_vars(fname, mode, hps):
  """where mode is one of 
    train, test, ktest, validation
  """
  assert mode in ('train', 'test', 'ktest', 'validation')
  if mode == 'train':
    epochs = None # cycle endlessly
  else:
    epochs = None #1
  fq = tf.train.string_input_producer([fname], num_epochs=epochs,
      name='{}-filename-queue'.format(mode))
  dat = read_and_decode(fq)

  shuffle = mode == 'train'
  if shuffle:
    # TODO: Possible capacity/min_after_deq needs to be set really high to keep
    # the data well mixed. Some users have 200+ products (and therefore as many sequences).
    # Maybe just do an offline one-time shuffle of the tfrecords file and be done with it?
    # Though I think the maybe_batch sampling thing should help a lot.
    dat = s_e_l._shuffle_inputs(dat, capacity=1000, min_after_dequeue=200,
        num_threads=1)

  batch_kwargs = {'dynamic_pad': True,
      'batch_size': hps.batch_size,
      'capacity': 1000,
  }
  batchfn = tf.train.batch
  # TODO: dynamic_pad doesn't work with shuffle_batch. arrrrrrgh.
  # shuffling was a huge reason to switch to this architecture.
  # deal with this later.
  # option 1: workaround like this https://github.com/tensorflow/tensorflow/issues/5147#issuecomment-271086206
  # option 2: store padded vectors? in which case, could just use Examples, rather
  # than SequenceExamples. That's probably stupid wasteful tho.
  if mode == 'train':
    batchfn = tf.train.maybe_batch
    keep = tf.random_uniform([]) < dat['weight']
    batch_kwargs['keep_input'] = keep
  elif mode in ('test', 'ktest'):
    # TODO: need to make sure code actually handles
    # these smaller batches properly.
    batch_kwargs['allow_smaller_final_batch'] = True

  batchdat = batchfn(dat, **batch_kwargs)
  # TODO: deal w weight in training at some level
  return batchdat


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  # TODO: is there any way to assign a name to each of the tensors
  # returned in these ops?
  ctxfeats = {
      'pid': tf.FixedLenFeature([], tf.int64),
      'aisleid': tf.FixedLenFeature([], tf.int64),
      'deptid': tf.FixedLenFeature([], tf.int64),
      'uid': tf.FixedLenFeature([], tf.int64),
      'seqlen': tf.FixedLenFeature([], tf.int64), # is this necessary?
      'weight': tf.FixedLenFeature([], tf.float32),
      # etc.
      # aisleid, deptid, (uid), seqlen, weight
  }
  nfeats = features.NFEATS
  seqfeats = {
      'label': tf.FixedLenSequenceFeature([], tf.float32),
      'lossmask': tf.FixedLenSequenceFeature([], tf.float32), # ???
      'feats': tf.FixedLenSequenceFeature([nfeats], tf.float32),
  }
  ctx, seq = tf.parse_single_sequence_example(
      serialized_example,
      context_features=ctxfeats,
      sequence_features=seqfeats,
      )
  feats = dict(ctx.items() + seq.items())
  return feats
