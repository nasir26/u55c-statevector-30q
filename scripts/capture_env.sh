#!/bin/bash
# Capture hardware / software versions for the configuration table (results/raw/env/). Read-only commands only.
source "${WS:-$PWD}/env.sh"
O=$WS/results/raw/env; mkdir -p $O
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; uname -a
  lscpu | grep -E 'Model name|Socket|Core\(s\) per socket|Thread\(s\) per core|NUMA node\(s\)|CPU MHz|L3'
  free -g | head -2; echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  lsb_release -d 2>/dev/null
} > $O/host.txt
numactl -H > $O/numa.txt 2>&1
{
  cat /opt/xilinx/xrt/version.json
  for b in 0000:41:00.1 0000:a1:00.1; do xbutil examine -d $b -r platform 2>&1 | grep -E 'XSA Name|Platform UUID|SC Version|Logic UUID|PCIe'; done
  ls /opt/xilinx/platforms; ls /tools/Xilinx/Vitis
} > $O/fpga.txt 2>&1
for e in fpga cpu; do $WS/envs/$e/bin/python - > $O/python_$e.txt 2>&1 <<'PY'
import sys, importlib
print("python", sys.version.split()[0])
for m in ("numpy", "scipy", "qiskit", "qiskit_aer", "qsimcirq", "cirq", "qulacs", "matplotlib", "pandas"):
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "n/a")
PY
done
gcc --version | head -1 > $O/compilers.txt
for f in $WS/builds/*/sv.xclbin $WS/builds/*/hbm1.xclbin; do [ -f $f ] && echo "$(sha256sum $f | cut -c1-64) $f"; done > $O/xclbin_sha256.txt
grep -h '^ENV' $WS/results/raw/gpu/*.log | sort -u > $O/gpu.txt
echo "captured into $O"
