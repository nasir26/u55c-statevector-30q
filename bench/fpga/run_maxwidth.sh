#!/bin/bash
# Largest width verification on one card: FPGA runs, park, then CPU references.
source "${WS:-$PWD}/env.sh"; source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; cd $WS
S=$WS/tmp/maxwidth_states; O=$WS/results/raw/phase5/sv_maxwidth_sv_e16w_m4.json; mkdir -p $S
# Bitstream the card is parked on afterwards (any low power design; the measurements used the v06 bitstream of
# the multi FPGA repository, u55c-multi-fpga-statevector/builds/v06). Set PARK to its path; the card and its BDF too.
PARK=${PARK:?set PARK to the parking bitstream}; BDF=${BDF:-0000:41:00.1}
echo "[$(date -Is)] start load=$(cat /proc/loadavg)"
taskset -c 0-15 envs/fpga/bin/python bench/fpga/sv_maxwidth.py --bdf $BDF --xclbin $WS/builds/sv_e16w_m4/sv.xclbin \
  --park-xclbin $PARK --ns 24,26,28,29,30 --state-dir $S --out $O || echo FPGA_PART_FAILED
echo "[$(date -Is)] fpga part done; references"
envs/fpga/bin/python bench/fpga/sv_maxwidth_ref.py --run $O --state-dir $S && rm -rf $S
echo "[$(date -Is)] MAXWIDTH_DONE"
