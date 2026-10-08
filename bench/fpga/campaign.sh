#!/bin/bash
# Phase 5 FPGA campaign. usage: campaign.sh <card bdf> <part>   (parts run sequentially on one card)
#   main     : flagship build, all families, n = 4..30, reps 10 (n <= 24) / 3 (n >= 26)
#   ablate   : design-space ablations at n = 26 on the flagship build
#   builds   : the same circuit set on every SV build (engines / PCs / port width)
#   energy   : sustained loops (>= 30 s) for steady-state board power, SV flagship and v07
#   jitter   : latency distribution, n = 4..16, 2000 runs, SV flagship and v07
#   suite    : fidelity suite on the flagship build (incl. analytic checks to n = 30)
source "${WS:-$PWD}/env.sh"; cd $WS
BDF=$1; PART=$2
O=$WS/results/raw/phase5; mkdir -p $O
# host-side hygiene: cap math-library threads and pin the process to the card's NUMA node (16 cores), so two
# concurrently running cards do not disturb each other's host-side timings
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
NODE=$(cat /sys/bus/pci/devices/$BDF/numa_node); FIRST=$(( NODE * 64 ))
P="taskset -c $FIRST-$(( FIRST + 15 )) $WS/envs/fpga/bin/python"
FLAG=${FLAG:-sv_e16w512}
XF=$WS/builds/$FLAG/sv.xclbin
V07=$WS/opt_kernel/opt_kernel/quantum_fpga_kernel_complex128.xclbin
log() { echo "[$(date -Is)] $BDF $PART: $*" | tee -a $O/fpga_campaign.log; }
log "start load=$(cat /proc/loadavg) vivado=$(pgrep -c vivado || true) pin=$FIRST-$(( FIRST + 15 )) threads=$OMP_NUM_THREADS"
case $PART in
main)
  $P bench/fpga/sv_bench.py --bdf $BDF --xclbin $XF --engines 16 --swaps 4 --families ghz,qft,rqc,vqe --rqc-depths 8,10,16,20 \
     --ns 4,6,8,10,12,14,16,18,20,22,24 --reps 10 --tag main --out $O/sv_main_small_${FLAG}.json
  $P bench/fpga/sv_bench.py --bdf $BDF --xclbin $XF --engines 16 --swaps 4 --families ghz,qft,rqc,vqe --rqc-depths 8,10,16,20 \
     --ns 26,28,29,30 --reps 3 --fid-max-n 0 --tag main --out $O/sv_main_large_${FLAG}.json
  ;;
ablate)
  N=26
  for pm in 1 2 4 8 16; do
    $P bench/fpga/sv_bench.py --bdf $BDF --xclbin $XF --engines 16 --swaps 4 --ns $N --reps 3 --p-max $pm --fid-max-n 0 \
       --tag ablate_pmax$pm --out $O/sv_ablate_pmax${pm}_${FLAG}.json; done
  for k in 9 10 11 12; do
    $P bench/fpga/sv_bench.py --bdf $BDF --xclbin $XF --engines 16 --swaps 4 --ns $N --reps 3 --k $k --fid-max-n 0 \
       --tag ablate_k$k --out $O/sv_ablate_k${k}_${FLAG}.json; done
  $P bench/fpga/sv_bench.py --bdf $BDF --xclbin $XF --engines 16 --swaps 4 --ns $N --reps 3 --fuse 0 --fid-max-n 0 --tag ablate_nofuse --out $O/sv_ablate_nofuse_${FLAG}.json
  $P bench/fpga/sv_bench.py --bdf $BDF --xclbin $XF --engines 16 --swaps 4 --ns $N --reps 3 --tiling 0 --fid-max-n 0 --tag ablate_notiling --out $O/sv_ablate_notiling_${FLAG}.json
  $P bench/fpga/sv_bench.py --bdf $BDF --xclbin $XF --engines 16 --swaps 4 --ns $N --reps 3 --fuse 0 --tiling 0 --fid-max-n 0 --tag ablate_none --out $O/sv_ablate_none_${FLAG}.json
  $P bench/fpga/sv_bench.py --bdf $BDF --xclbin $XF --engines 16 --swaps 4 --ns $N --reps 3 --layout modulo --fid-max-n 0 --tag ablate_modulo --out $O/sv_ablate_modulo_${FLAG}.json
  $P bench/fpga/sv_bench.py --bdf $BDF --xclbin $XF --engines 16 --swaps 4 --ns $N --reps 3 --no-device-swap --fid-max-n 0 --tag ablate_hostswap --out $O/sv_ablate_hostswap_${FLAG}.json
  ;;
builds)
  for b in sv_e8:8 sv_e16:16 sv_e16w:16 sv_e16w512:16 sv_e16w_m4:16; do
    nm=${b%%:*}; ne=${b##*:}
    [ -f $WS/builds/$nm/sv.xclbin ] || { log "skip $nm (no xclbin)"; continue; }
    ns=20,22,24,26,28; case $nm in sv_e16w|sv_e16w512|sv_e16w_m4) ns=20,22,24,26,28,30;; esac
    $P bench/fpga/sv_bench.py --bdf $BDF --xclbin $WS/builds/$nm/sv.xclbin --engines $ne --swaps 4 --ns $ns --reps 3 --fid-max-n 0 \
       --tag builds --out $O/sv_builds_${nm}.json
  done
  ;;
energy)
  $P bench/fpga/sv_bench.py --bdf $BDF --xclbin $XF --engines 16 --swaps 4 --ns 20,24,26,28,30 --reps 1 --fid-max-n 0 --energy-loop 30 \
     --tag energy --out $O/sv_energy_${FLAG}.json
  $P bench/fpga/v07_energy.py --bdf $BDF --xclbin $V07 --ns 20,24 --loop 30 --out $O/v07_energy.json
  ;;
jitter)
  $P bench/latency_jitter.py --platform sv --bdf $BDF --xclbin $XF --engines 16 --swaps 4 --runs 2000 --out $O/jitter_sv_${FLAG}.json
  $P bench/latency_jitter.py --platform v07 --bdf $BDF --xclbin $V07 --runs 2000 --out $O/jitter_v07.json
  # CPU latency distribution on the same host core set (single process, 8 threads), for the determinism comparison
  CPUP="taskset -c $FIRST-$(( FIRST + 15 )) $WS/envs/cpu/bin/python"
  $CPUP bench/latency_jitter.py --platform aer --threads 8 --runs 2000 --out $O/jitter_cpu_aer8.json
  $CPUP bench/latency_jitter.py --platform qulacs --runs 2000 --out $O/jitter_cpu_qulacs.json
  ;;
suite)
  $P tests/run_suite_sv.py $BDF $XF 16 4 $WS/results/raw/phase4/sv_functional_${FLAG}.json 24,26,28,30
  ;;
esac
log "done load=$(cat /proc/loadavg)"
