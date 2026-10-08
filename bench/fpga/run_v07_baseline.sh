#!/bin/bash
source "${WS:-$PWD}/env.sh"; cd $WS
P=envs/fpga/bin/python
$P bench/fpga/v07_bench.py --bdf 0000:41:00.1 --families ghz,qft,rqc,vqe --ns 4,8,12,16,20,24 --reps 3 --out results/raw/phase1/v07_baseline_grid.json
$P bench/fpga/v07_bench.py --bdf 0000:41:00.1 --families ghz --ns 26,28 --reps 1 --fid-max-n 28 --out results/raw/phase1/v07_baseline_ghz_large.json
echo BASELINE_DONE
