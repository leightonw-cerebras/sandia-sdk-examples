#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./src/layout.csl --fabric-dims=8,3 \
--fabric-offsets=4,1 --params=M:4,K:4,N:6 -o out --memcpy --channels 1
cs_python run.py --name out
