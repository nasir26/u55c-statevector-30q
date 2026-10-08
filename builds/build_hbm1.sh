#!/bin/bash
# usage: build_hbm1.sh <name> <width 256|512> <num_CUs> <pcs_per_CU>
source "${WS:-$PWD}/env.sh"; vitis_env
NAME=$1; W=$2; NE=$3; PW=$4
B=$WS/builds/$NAME; mkdir -p $B; cd $B
v++ -c -t hw --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -k hbm1 -D HBM1_W=$W --hls.clock 300000000:hbm1 \
  --temp_dir $B/_xc --report_dir $B/reports --log_dir $B/logs -o $B/hbm1.xo $WS/kernels/hbm_stream/hbm1.cpp || exit 1
{ echo "[connectivity]"; echo "nk=hbm1:$NE"
  for i in $(seq 1 $NE); do pc=$(( (i-1)*PW )); if [ $PW -gt 1 ]; then echo "sp=hbm1_$i.p:HBM[$pc:$((pc+PW-1))]"; else echo "sp=hbm1_$i.p:HBM[$pc]"; fi
    echo "slr=hbm1_$i:SLR$(( (i-1) * 3 / NE ))"; done; } > link.cfg
v++ -l -t hw --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config link.cfg --kernel_frequency 300 -j 12 \
  --temp_dir $B/_x --report_dir $B/reports --log_dir $B/logs -o $B/hbm1.xclbin $B/hbm1.xo || exit 1
sha256sum hbm1.xclbin > hbm1.xclbin.sha256; echo BUILD_OK
