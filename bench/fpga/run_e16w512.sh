#!/bin/bash
source "${WS:-$PWD}/env.sh"; cd $WS
envs/fpga/bin/python tests/run_suite_sv.py 0000:41:00.1 builds/sv_e16w512/sv.xclbin 16 4 results/raw/phase4/sv_functional_sv_e16w512.json 24,26,28,30 > logs/phase4_sv_e16w512_suite.log 2>&1
envs/fpga/bin/python bench/fpga/sv_bench.py --bdf 0000:41:00.1 --xclbin builds/sv_e16w512/sv.xclbin --engines 16 --swaps 4 --ns 20,22,24,26,28,30 --reps 3 --fid-max-n 0 --tag builds --out results/raw/phase5/sv_builds_sv_e16w512.json > logs/sv_builds_e16w512.log 2>&1
echo E16W512_DONE
