#!/bin/bash
# Optional cross-check of the revised FPGA simulator on another host with one Alveo U55C.
# The paper's FPGA and CPU numbers come from the original host; this only confirms the card behaves identically.
# Usage: bench/fpga/crosscheck_other_host.sh <card BDF, e.g. 0000:3b:00.1>
# Requires: platform xilinx_u55c_gen3x16_xdma_3_202210_1 (shell xilinx_u55c_gen3x16_xdma_base_3), XRT 2.16, Python 3.8
# env with pyxrt (create with: envs/bin/micromamba create -p envs/fpga python=3.8 numpy=1.24 scipy matplotlib pandas;
# envs/fpga/bin/pip install qiskit==1.2.4 qiskit-aer==0.15.1).
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BDF=${1:?usage: crosscheck_other_host.sh <BDF>}
source /opt/xilinx/xrt/setup.sh > /dev/null
export WS PYTHONPATH=/opt/xilinx/xrt/python:${PYTHONPATH:-} PYTHONNOUSERSITE=1 TMPDIR=$WS/tmp OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
H=$(hostname -s); O=$WS/results/raw/crosscheck_$H; mkdir -p $O $WS/tmp
echo "== checking platform"
xbutil examine -d $BDF -r platform | tee $O/platform.txt | grep -E 'XSA Name|Logic UUID'
grep -q 'xilinx_u55c_gen3x16_xdma_base_3' $O/platform.txt || { echo "different shell: rebuild the xclbins for this platform"; exit 1; }
cat /opt/xilinx/xrt/version.json > $O/xrt_version.json
X=$WS/builds/sv_e16w_m4/sv.xclbin
sha256sum $X | tee $O/xclbin.sha256
grep -q "$(cut -c1-64 $O/xclbin.sha256)" $WS/results/raw/env/xclbin_sha256.txt && echo "xclbin hash matches the original host"
P=$WS/envs/fpga/bin/python
echo "== functional suite"
$P $WS/tests/run_suite_sv.py $BDF $X 16 4 $O/sv_functional_sv_e16w_m4.json 24,26,28,30
echo "== timing subset"
$P $WS/bench/fpga/sv_bench.py --bdf $BDF --xclbin $X --engines 16 --swaps 4 --ns 20,24,28 --reps 3 --fid-max-n 0 \
   --tag crosscheck --out $O/sv_crosscheck_sv_e16w_m4.json
echo "CROSSCHECK_DONE: compare $O with results/raw/phase5/sv_main_*_sv_e16w_m4.json (analysis/crosscheck.py)"
