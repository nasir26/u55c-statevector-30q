#!/bin/bash
# usage: link_sv.sh <name> <num_CUs> [pc_stride] [num_swap_CUs] [pcs_per_CU]
#   engine CU i -> HBM[i*stride], CUs spread over SLR0..2; swap CUs: both ports HBM[0:31], SLR0
source "${WS:-$PWD}/env.sh"; vitis_env
NAME=$1; NE=$2; STRIDE=${3:-$((32/NE))}; NS=${4:-0}; W=${5:-1}
B=$WS/builds/$NAME; mkdir -p $B; cd $B
{ echo "[connectivity]"; echo "nk=sv_engine:$NE"
  for i in $(seq 1 $NE); do pc=$(( (i-1)*STRIDE )); if [ $W -gt 1 ]; then echo "sp=sv_engine_$i.st:HBM[$pc:$((pc+W-1))]"; else echo "sp=sv_engine_$i.st:HBM[$pc]"; fi
    s=$(( (i-1) * 3 / NE )); echo "slr=sv_engine_$i:SLR$s"; done
  if [ $NS -gt 0 ]; then echo "nk=sv_swap:$NS"
    for i in $(seq 1 $NS); do echo "sp=sv_swap_$i.a:HBM[0:31]"; echo "sp=sv_swap_$i.b:HBM[0:31]"; echo "slr=sv_swap_$i:SLR0"; done; fi; } > link.cfg
XOS="${SV_XO:-$WS/builds/sv_xo/sv_engine.xo}"; [ $NS -gt 0 ] && XOS="$XOS $WS/builds/sv_xo/sv_swap.xo"
v++ -l -t hw --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config link.cfg --kernel_frequency 300 -j 16 \
  --temp_dir $B/_x --report_dir $B/reports --log_dir $B/logs -o $B/sv.xclbin $XOS || exit 1
sha256sum sv.xclbin > sv.xclbin.sha256; echo BUILD_OK
