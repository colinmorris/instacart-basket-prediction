#!/usr/bin/env bash
./libfm/bin/libFM -task c -train vectors/train.libfm -test vectors/test.libfm -out predictions.out -meta groups.txt -method sgda -validation vectors/test.libfm -learn_rate 0.0001 $@
