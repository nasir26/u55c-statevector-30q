#!/bin/bash
source "${WS:-$PWD}/env.sh"; vitis_env
B=$WS/builds/sv_xo; cd $B
v++ -c -t hw --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -k sv_swap --hls.clock 300000000:sv_swap \
  --temp_dir $B/_xs --report_dir $B/reports_swap --log_dir $B/logs_swap -o $B/sv_swap.xo $WS/kernels/sv/sv_swap.cpp
