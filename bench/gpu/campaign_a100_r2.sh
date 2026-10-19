#!/bin/bash
# Second A100 campaign (2026-10-14): the configurations whose first jobs were cancelled externally on 2026-10-06.
# Same script, circuits, seeds and package versions as bench/gpu/campaign_a100.sh.
R=$WS/bench/gpu/run_remote.sh
N=gpu-node; G=gpu:a100:1
NS=4,6,8,10,12,14,16,18,20,22,24,26,28,29,30,31,32,33
$R a100_aer_double_cusv $N $G 04:00:00 -- --backend aer_gpu --precision double --custatevec 1 --ns 4,8,12,16,20,24,26,28,30,32 --reps 3
$R a100_cudaq_fp32      $N $G 04:00:00 -- --backend cudaq   --precision single --ns $NS --reps 5
$R a100_cudaq_fp64      $N $G 04:00:00 -- --backend cudaq   --precision double --ns 4,8,12,16,20,24,26,28,30,32 --reps 3
echo CAMPAIGN_R2_DONE
