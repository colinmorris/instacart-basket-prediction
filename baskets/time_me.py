import time
from contextlib import contextmanager
import tensorflow as tf

@contextmanager
def time_me(what_i_did='Finished'):
  t0 = time.time()
  yield
  t1 = time.time()
  msg = '{} in {:.1f}s'.format(t1-t0)
  tf.logging.info(msg)
