#!/usr/bin/env bash

# TODO: try the save_model flag
# Also, try rlog
./libfm/bin/libFM -task c \
  -train vectors/train.libfm -test vectors/test.libfm \
  -out predictions.out \
  -meta groups.txt \
  $@
