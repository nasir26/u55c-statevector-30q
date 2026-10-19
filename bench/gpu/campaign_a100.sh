#!/bin/bash
# Full GPU campaign on gpu-node (A100-80GB). One job per backend config (Slurm serialises them on the single GPU).
R=$WS/bench/gpu/run_remote.sh
N=gpu-node; G=gpu:a100:1
NS=4,6,8,10,12,14,16,18,20,22,24,26,28,29,30,31,32,33
$R a100_aer_single_cusv  $N $G 06:00:00 -- --backend aer_gpu --precision single --custatevec 1 --ns $NS --reps 5
$R a100_cudaq_fp32        $N $G 06:00:00 -- --backend cudaq   --precision single --ns $NS --reps 5
$R a100_aer_single_nocusv $N $G 06:00:00 -- --backend aer_gpu --precision single --custatevec 0 --ns $NS --reps 5
$R a100_aer_double_cusv  $N $G 06:00:00 -- --backend aer_gpu --precision double --custatevec 1 --ns 4,8,12,16,20,24,26,28,30,32 --reps 3
$R a100_cudaq_fp64        $N $G 06:00:00 -- --backend cudaq   --precision double --ns 4,8,12,16,20,24,26,28,30,32 --reps 3
echo CAMPAIGN_DONE
