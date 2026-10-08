#!/bin/bash
# usage: build_hbm_variant.sh <variant> <src.cpp> [bram|uram]   -> builds/hbm_stream_<variant>/
source "${WS:-$PWD}/env.sh"; vitis_env
V=$1; SRC=$WS/kernels/hbm_stream/$2; BUF=${3:-auto}
B=$WS/builds/hbm_stream_$V; mkdir -p $B; cd $B
PF=xilinx_u55c_gen3x16_xdma_3_202210_1
echo "config_interface -m_axi_buffer_impl $BUF" > $B/pre.tcl
v++ -c -t hw --platform $PF -k hbm_stream --hls.clock 300000000:hbm_stream --hls.pre_tcl $B/pre.tcl \
    --temp_dir $B/_x --report_dir $B/reports --log_dir $B/logs -o $B/hbm_stream.xo $SRC || exit 1
v++ -l -t hw --platform $PF --config $WS/kernels/hbm_stream/hbm_stream.cfg --kernel_frequency 300 -j 16 \
    --temp_dir $B/_x --report_dir $B/reports --log_dir $B/logs -o $B/hbm_stream.xclbin $B/hbm_stream.xo || exit 1
sha256sum $B/hbm_stream.xclbin > $B/hbm_stream.xclbin.sha256
echo BUILD_OK
