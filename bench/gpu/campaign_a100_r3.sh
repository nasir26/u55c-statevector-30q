#!/bin/bash
# Third A100 campaign (2026-10-14): CUDA-Q for the families the r2 jobs did not reach (r2 fp32 killed at QFT n = 32,
# r2 fp64 cancelled at QFT n = 32). Widths capped at 30 qubits to stay within node memory and the 4 h partition limit.
R=$WS/bench/gpu/run_remote.sh
N=gpu-node; G=gpu:a100:1
NS=4,6,8,10,12,14,16,18,20,22,24,26,28,29,30
$R a100_cudaq_fp32_r3 $N $G 02:00:00 -- --backend cudaq --precision single --families rqc,vqe --ns $NS --reps 5
$R a100_cudaq_fp64_r3 $N $G 02:00:00 -- --backend cudaq --precision double --families rqc,vqe --ns 4,8,12,16,20,24,26,28,30 --reps 3
echo CAMPAIGN_R3_DONE
