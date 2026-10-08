#!/bin/bash
# CPU baseline campaign. Run only when no builds or other heavy jobs are active (checked and logged).
source "${WS:-$PWD}/env.sh"; cd $WS
O=$WS/results/raw/phase5; mkdir -p $O
P=$WS/envs/cpu/bin/python
log() { echo "[$(date -Is)] $*" | tee -a $O/cpu_campaign.log; }
log "load: $(cat /proc/loadavg); vivado/v++ procs: $(pgrep -fc 'vivado|v\+\+' || true); governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
# 1. memory bandwidth ceilings
OMP_NUM_THREADS=128 OMP_PLACES=cores OMP_PROC_BIND=spread numactl --interleave=all tmp/triad $((1<<29)) > $O/cpu_triad_128c_interleave.json
OMP_NUM_THREADS=256 OMP_PLACES=threads OMP_PROC_BIND=spread numactl --interleave=all tmp/triad $((1<<29)) > $O/cpu_triad_256t_interleave.json
OMP_NUM_THREADS=64 OMP_PLACES=cores OMP_PROC_BIND=close numactl --cpunodebind=0 --membind=0 tmp/triad $((1<<28)) > $O/cpu_triad_64c_node0.json
log "triad: $(cat $O/cpu_triad_*.json | tr '\n' ' ')"
NS=4,8,12,16,20,22,24,26,28,30,32
# 2. Qiskit Aer, all cores, interleaved
for prec in single double; do for fus in 1 0; do
  OMP_PROC_BIND=spread OMP_PLACES=cores numactl --interleave=all $P bench/cpu/cpu_bench.py --backend aer --threads 128 \
     --precision $prec --fusion $fus --ns $NS --reps 3 --out $O/cpu_aer_128t_${prec}_f${fus}.json >> $O/cpu_campaign.log 2>&1
  log "aer $prec fusion=$fus done"; done; done
# 3. Aer one socket (64 cores, local memory)
OMP_PROC_BIND=close OMP_PLACES=cores numactl --cpunodebind=0 --membind=0 $P bench/cpu/cpu_bench.py --backend aer --threads 64 \
   --precision single --fusion 1 --ns 4,8,12,16,20,24,26,28,30 --reps 3 --out $O/cpu_aer_64t_node0_single_f1.json >> $O/cpu_campaign.log 2>&1
log "aer node0 done"
# 4. qsim (single precision) and Qulacs (double)
OMP_PROC_BIND=spread OMP_PLACES=cores numactl --interleave=all $P bench/cpu/cpu_bench.py --backend qsim --threads 128 \
   --ns 4,8,12,16,20,22,24,26,28,30 --reps 3 --out $O/cpu_qsim_128t.json >> $O/cpu_campaign.log 2>&1; log "qsim done"
OMP_NUM_THREADS=128 OMP_PROC_BIND=spread OMP_PLACES=cores numactl --interleave=all $P bench/cpu/cpu_bench.py --backend qulacs \
   --threads 128 --ns 4,8,12,16,20,22,24,26,28,30 --reps 3 --out $O/cpu_qulacs_128t.json >> $O/cpu_campaign.log 2>&1; log "qulacs done"
# 5. legacy single-thread NumPy (complex64), corrected vectorised implementation
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 numactl --cpunodebind=0 --membind=0 $P bench/cpu/cpu_bench.py --backend numpy1t \
   --ns 4,8,12,16,20,22,24,26,28 --reps 3 --out $O/cpu_numpy1t.json >> $O/cpu_campaign.log 2>&1; log "numpy done"
log "CPU_CAMPAIGN_DONE load: $(cat /proc/loadavg)"
