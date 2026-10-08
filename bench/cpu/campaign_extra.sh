#!/bin/bash
# Top-up CPU campaign: brings every CPU point to 10 timed repetitions (n <= 24) and 3 (n >= 26), matching the FPGA protocol.
# Each file's rep 0 is again a warm-up and is excluded; analysis/load.py pools files of the same configuration.
# Run only when no FPGA timing runs or builds are active (checked and logged).
source "${WS:-$PWD}/env.sh"; cd $WS
O=$WS/results/raw/phase5; P=$WS/envs/cpu/bin/python; L=$O/cpu_campaign_extra.log
log() { echo "[$(date -Is)] $*" | tee -a $L; }
log "load: $(cat /proc/loadavg); vivado/v++: $(pgrep -fc 'vivado|v\+\+' || true); fpga benches: $(pgrep -fc 'sv_bench|latency_jitter|v07_bench' || true)"
SMALL=4,8,12,16,20,22,24
I="numactl --interleave=all"
run() {  # run <tag> <ns> <reps> <env/prefix...> -- <bench args>
  local tag=$1 ns=$2 reps=$3; shift 3
  "$@" --ns $ns --reps $reps --out $O/${tag}.json >> $L 2>&1; log "$tag done"
}
for sz in small large; do
  if [ $sz = small ]; then NS=$SMALL; R=9; else NS=26,28,30; R=2; fi
  for prec in single double; do for fus in 1 0; do
    NSA=$NS; [ $sz = large ] && NSA=26,28,30,32
    run cpu_aer_128t_${prec}_f${fus}_extra_$sz $NSA $R env OMP_PROC_BIND=spread OMP_PLACES=cores $I $P bench/cpu/cpu_bench.py \
        --backend aer --threads 128 --precision $prec --fusion $fus
  done; done
  run cpu_aer_64t_node0_single_f1_extra_$sz $NS $R env OMP_PROC_BIND=close OMP_PLACES=cores numactl --cpunodebind=0 --membind=0 \
      $P bench/cpu/cpu_bench.py --backend aer --threads 64 --precision single --fusion 1
  run cpu_qsim_128t_extra_$sz $NS $R env OMP_PROC_BIND=spread OMP_PLACES=cores $I $P bench/cpu/cpu_bench.py --backend qsim --threads 128
  run cpu_qulacs_128t_extra_$sz $NS $R env OMP_NUM_THREADS=128 OMP_PROC_BIND=spread OMP_PLACES=cores $I $P bench/cpu/cpu_bench.py \
      --backend qulacs --threads 128
done
log "CPU_EXTRA_DONE load: $(cat /proc/loadavg)"
