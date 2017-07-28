import sys

import hypers
import dataset

tag = sys.argv[1]
hps = hypers.hps_for_tag(tag)

ds = dataset.Dataset('test', hps)

