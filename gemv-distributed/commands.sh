#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./src/layout.csl --fabric-dims=11,6 \
--fabric-offsets=4,1 --params=kernel_x_dim:4,kernel_y_dim:4,M:32,N:16 \
--params=MEMCPYH2D_DATA_1_ID:0 \
--params=MEMCPYH2D_DATA_2_ID:1 \
--params=MEMCPYD2H_DATA_1_ID:2 \
-o out --memcpy --channels 1
cs_python run.py --name out
