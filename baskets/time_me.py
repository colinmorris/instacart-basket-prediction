import time
import sys
from contextlib import contextmanager
import tensorflow as tf

DEFAULT_MODE = 'tflog'

def set_default_mode(mode):
  global DEFAULT_MODE
  DEFAULT_MODE = mode

@contextmanager
def time_me(what_i_did='Finished', mode=None):
  if mode is None:
    mode = DEFAULT_MODE
  t0 = time.time()
  yield
  t1 = time.time()
  msg = '{} in {:.1f}s'.format(what_i_did, t1-t0)
  if mode == 'tflog':
    tf.logging.info(msg)
  elif mode == 'print':
    print msg
  elif mode == 'stderr':
    sys.stderr.write(msg+'\n')
  else:
    assert False, 'Unrecognized mode {}'.format(mode)
