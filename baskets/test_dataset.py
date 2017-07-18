import pytest

from baskets.dataset import BasketDataset
from baskets import hypers
from baskets.test_helpers import sess

@pytest.fixture
def vectors_path():
  return 'unit_tests.tfrecords'

def test_dataset_creation(vectors_path):
  hps = hypers.get_default_hparams()
  ds = BasketDataset(hps, vectors_path)

def test_data_fetch(vectors_path, sess):
  hps = hypers.get_default_hparams()
  batch_size = 10
  hps.batch_size = batch_size
  ds = BasketDataset(hps, vectors_path)
  vdict = ds.model_input_dict()
  tdict = sess.run(vdict)

  seqlens = tdict['seqlen']
  assert seqlens.shape == (batch_size,)
  labels = tdict['labels']
  label_shape = labels.shape
  assert len(label_shape) == 2
  assert label_shape[0] == batch_size
  padlen = label_shape[1]
  # (Can only see print msgs with pytest -s)
  print "Batch padded to length {}".format(padlen)
  assert 2 <= padlen <= 100
  lossmask = tdict['lossmask']
  assert lossmask.shape == label_shape



if __name__ == '__main__':
  pytest.main([__file__])
