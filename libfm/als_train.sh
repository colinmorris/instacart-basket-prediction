#!/usr/bin/env bash
./libfm/bin/libFM -task c -train vectors/train.libfm -test vectors/test.libfm -out predictions.out -meta groups.txt -method als -regular 0,100,1000 -init_stdev 0.01 -iter 80
