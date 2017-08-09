#!/usr/bin/env bash

# TODO: try the save_model flag
# NB: -init_stdev 0.01 seems to work well in many cases.
./libfm/bin/libFM -task c \
  -train vectors/train.libfm -test vectors/test.libfm \
  -out predictions.out \
  $@
