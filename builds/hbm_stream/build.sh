#!/bin/bash
# Build the HBM streaming microbenchmark. Usage: build.sh [compile|link]
source "${WS:-$PWD}/env.sh"; vitis_env
B=$WS/builds/hbm_stream; K=$WS/kernels/hbm_stream; cd $B
PF=xilinx_u55c_gen3x16_xdma_3_202210_1
if [ "$1" != link ]; then
  v++ -c -t hw --platform $PF -k hbm_stream --hls.clock 300000000:hbm_stream \
      --temp_dir $B/_x --report_dir $B/reports --log_dir $B/logs -o $B/hbm_stream.xo $K/hbm_stream.cpp || exit 1
fi
if [ "$1" != compile ]; then
  v++ -l -t hw --platform $PF --config $K/hbm_stream.cfg --kernel_frequency 300 -j 16 \
      --temp_dir $B/_x --report_dir $B/reports --log_dir $B/logs -o $B/hbm_stream.xclbin $B/hbm_stream.xo || exit 1
fi
