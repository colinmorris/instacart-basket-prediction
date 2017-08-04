import sys

from feature_spec import FeatureSpec

try:
  vecpath = sys.argv[1]
except IndexError:
  vecpath = 'vectors/test.libfm'

f = open(vecpath)
spec = FeatureSpec.all_features_spec()

def poke():
  l = f.readline()
  fd = spec.parse_libfm_line(l)
  return fd

fd = poke()
print fd
