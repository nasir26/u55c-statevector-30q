#!/bin/bash
source "${WS:-$PWD}/env.sh"; vitis_env
B=$WS/builds/sv_xo; cd $B
v++ -c -t hw --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -k sv_engine --hls.clock 300000000:sv_engine \
  --temp_dir $B/_x --report_dir $B/reports --log_dir $B/logs -o $B/sv_engine.xo $WS/kernels/sv/sv_engine.cpp
