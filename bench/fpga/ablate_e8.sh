#!/bin/bash
source "${WS:-$PWD}/env.sh"; cd $WS
O=$WS/results/raw/phase5; X=$WS/builds/sv_e8/sv.xclbin; B=0000:a1:00.1; N=26
P=$WS/envs/fpga/bin/python
run() { $P bench/fpga/sv_bench.py --bdf $B --xclbin $X --engines 8 --swaps 4 --ns $N --reps 3 --fid-max-n 0 "$@"; }
for pm in 1 2 4 8; do run --p-max $pm --tag e8_pmax$pm --out $O/sv_ablate_e8_pmax$pm.json; done
for k in 9 10 11 12; do run --k $k --tag e8_k$k --out $O/sv_ablate_e8_k$k.json; done
run --fuse 0 --tag e8_nofuse --out $O/sv_ablate_e8_nofuse.json
run --tiling 0 --tag e8_notiling --out $O/sv_ablate_e8_notiling.json
run --fuse 0 --tiling 0 --tag e8_none --out $O/sv_ablate_e8_none.json
run --layout modulo --tag e8_modulo --out $O/sv_ablate_e8_modulo.json
run --no-device-swap --tag e8_hostswap --out $O/sv_ablate_e8_hostswap.json
echo ABLATE_E8_DONE
