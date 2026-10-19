#!/bin/bash
# usage: run_remote.sh <tag> <node> <gres> <time> -- <gpu_bench.py args...>
# Ships code via stdin, runs in node-local /tmp (removed on exit), collects stdout into results/raw/gpu/<tag>.log
source "${WS:-$PWD}/env.sh"
TAG=$1; NODE=$2; GRES=$3; TLIM=$4; shift 5
OUT=$WS/results/raw/gpu; mkdir -p $OUT
ARGS="$*"
STAGE=$(mktemp -d $WS/tmp/gpustage.XXXX)
SCRIPT=${RR_SCRIPT:-gpu_bench.py}
PIPS=${RR_PIP:-"'qiskit==1.2.4' 'qiskit-aer-gpu==0.15.1' 'cudaq==0.12.0'"}
cp $WS/circuits/generators.py $WS/bench/gpu/gpu_bench.py $WS/bench/gpu/gpu_micro.py $WS/envs/bin/micromamba $STAGE/
tar czf - -C $STAGE . | srun --export=NONE -p gpu -w $NODE --gres=$GRES -c 16 -t $TLIM --chdir=/tmp --input=0 /bin/bash -c "
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
set -e
D=/tmp/jsc_user_\$SLURM_JOB_ID; mkdir -p \$D; trap 'rm -rf \$D' EXIT; cd \$D; tar xzf -
export HOME=\$D MAMBA_ROOT_PREFIX=\$D/mm PIP_CACHE_DIR=\$D/pipcache XDG_CACHE_HOME=\$D/cache PYTHONNOUSERSITE=1
./micromamba create -q -y -p \$D/env -c conda-forge python=3.11 > /dev/null
\$D/env/bin/pip install -q $PIPS > pip.log 2>&1 || (tail -20 pip.log; exit 1)
echo ENV \$(hostname) \$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) \$(\$D/env/bin/pip freeze 2>/dev/null | grep -iE '^(qiskit|qiskit-aer-gpu|cudaq|cupy|cuquantum)' | tr '\\n' ' ')
\$D/env/bin/python $SCRIPT $ARGS
" > $OUT/$TAG.log 2>&1
rc=$?; rm -rf $STAGE
grep '^RESULT ' $OUT/$TAG.log | sed 's/^RESULT //' > $OUT/$TAG.jsonl
echo "rc=$rc results=$(wc -l < $OUT/$TAG.jsonl)"; grep '^ENV' $OUT/$TAG.log
