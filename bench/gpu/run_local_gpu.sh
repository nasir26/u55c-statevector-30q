#!/bin/bash
# GPU campaign on a local NVIDIA GPU (written for the RTX A4000 system used in the paper; works on any CUDA 12 GPU).
# Everything is installed inside the workspace: envs/gpu (Python 3.11 via the bundled micromamba), caches under cache/.
# Usage:  bench/gpu/run_local_gpu.sh <tag>   e.g.  bench/gpu/run_local_gpu.sh a4000
# Results: results/raw/gpu/<tag>_*.jsonl (+ .log), env capture in results/raw/env/gpu_<tag>.txt
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG=${1:?usage: run_local_gpu.sh <tag>}
export TMPDIR=$WS/tmp PIP_CACHE_DIR=$WS/cache/pip XDG_CACHE_HOME=$WS/cache PYTHONNOUSERSITE=1 MAMBA_ROOT_PREFIX=$WS/envs/mamba
mkdir -p $TMPDIR $WS/results/raw/gpu $WS/results/raw/env
OUT=$WS/results/raw/gpu
PY=$WS/envs/gpu/bin/python

# 1. environment (once)
if [ ! -x $PY ]; then
  HOME=$WS/tmp/fakehome $WS/envs/bin/micromamba --no-rc create -y -q -p $WS/envs/gpu -c conda-forge python=3.11
  $PY -m pip install -q 'qiskit==1.2.4' 'qiskit-aer-gpu==0.15.1' 'cudaq==0.12.0' cupy-cuda12x cuquantum-python-cu12
fi

# 2. environment capture
{
  date -Is; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
  nvidia-smi -q | grep -iE 'CUDA Version|Persistence|Compute Mode|Product Name'
  lscpu | grep -E 'Model name|Socket|Core\(s\) per socket'; free -g | head -2
  $PY -m pip freeze | grep -iE '^(qiskit|qiskit-aer-gpu|cudaq|cupy|cuquantum)'
} > $WS/results/raw/env/gpu_${TAG}.txt 2>&1

cd $WS/bench/gpu
cp $WS/circuits/generators.py . 2>/dev/null || true
run() {   # run <name> <args...>: RESULT lines -> jsonl
  local name=$1; shift
  $PY gpu_bench.py "$@" > $OUT/${TAG}_${name}.log 2>&1 || echo "WARN: $name exited with $?"
  grep '^RESULT ' $OUT/${TAG}_${name}.log | sed 's/^RESULT //' > $OUT/${TAG}_${name}.jsonl
  echo "$name: $(wc -l < $OUT/${TAG}_${name}.jsonl) records"
}
# sizes: 16 GB device memory holds 30 qubits in complex64 (8 GB) and 29 in complex128 (8 GB); larger sizes are attempted
# and reported as failures if they do not fit (do not remove them: the failure itself is a result)
NS32=4,6,8,10,12,14,16,18,20,22,24,26,28,29,30
NS64=4,8,12,16,20,24,26,28,29,30
run aer_single_cusv   --backend aer_gpu --precision single --custatevec 1 --ns $NS32 --reps 5
run cudaq_fp32        --backend cudaq   --precision single --ns $NS32 --reps 5
run aer_single_nocusv --backend aer_gpu --precision single --custatevec 0 --ns $NS32 --reps 5
run aer_double_cusv   --backend aer_gpu --precision double --custatevec 1 --ns $NS64 --reps 3
run cudaq_fp64        --backend cudaq   --precision double --ns $NS64 --reps 3
# energy: >= 30 s loops at selected sizes (nvidia-smi power every 100 ms)
run energy_cudaq_fp32 --backend cudaq --precision single --ns 20,24,26,28,30 --reps 1 --energy-loop 30
run energy_aer_single --backend aer_gpu --precision single --custatevec 1 --ns 20,24,26,28,30 --reps 1 --energy-loop 30
# latency distribution at small n (2000 runs each), CUDA-Q fp32
run jitter_cudaq_fp32 --backend cudaq --precision single --families ghz,rqc --ns 4,8,12,16 --reps 2000 --fid-max-n 0
# device-level microbenchmark (cuStateVec gate sweep, device copy bandwidth)
$PY gpu_micro.py > $OUT/${TAG}_micro.log 2>&1 || echo "WARN: micro exited with $?"
grep '^RESULT ' $OUT/${TAG}_micro.log | sed 's/^RESULT //' > $OUT/${TAG}_micro.jsonl
echo "GPU_CAMPAIGN_DONE $(date -Is)"
