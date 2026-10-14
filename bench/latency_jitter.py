"""Latency distribution (determinism study): repeat small circuits many times, record per-run times.

platform: sv (new FPGA kernel), v07 (baseline FPGA), aer (CPU, threads), qulacs (CPU), numpy1t (CPU).
Records every run's T_e2e and, where the platform separates it, T_kernel; summary percentiles are computed in
analysis. Warm-up runs (first 20) are recorded but flagged.
"""
import uuid  # noqa: F401  (before any XRT device init)
import argparse, json, os, sys, time
import numpy as np

WS = os.environ.get("WS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path[:0] = [f"{WS}/host/sv", f"{WS}/host/v07", f"{WS}/circuits", f"{WS}/bench/cpu"]
from generators import make, to_tuples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--bdf"); ap.add_argument("--xclbin"); ap.add_argument("--engines", type=int, default=16)
    ap.add_argument("--swaps", type=int, default=4); ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--families", default="ghz,rqc"); ap.add_argument("--ns", default="4,8,12,16")
    ap.add_argument("--runs", type=int, default=2000); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.platform == "sv":
        from runtime import SVRuntime
        rt = SVRuntime(a.bdf, a.xclbin, a.engines, a.swaps)
        def run(qc):
            r = rt.run(qc, shots=1024, return_state=False); return r["timing"]["T_e2e"], r["timing"]["T_kernel"]
    elif a.platform == "v07":
        import fpga_simulator_v07 as v07
        eng = v07.QuantumFPGACircuit(a.xclbin, device_index=a.bdf)
        def run(qc):
            t0 = time.perf_counter(); eng.run_circuit(qc.num_qubits, to_tuples(qc), shots=1024, validate=False)
            return time.perf_counter() - t0, eng._t_launch_us / 1e6
    elif a.platform in ("aer", "qulacs", "numpy1t"):
        import cpu_bench
        def run(qc):
            t0 = time.perf_counter()
            if a.platform == "aer":
                _, x = cpu_bench.run_aer(qc, a.threads, "single", True); k = x["T_run_s"]
            elif a.platform == "qulacs":
                _, x = cpu_bench.run_qulacs(to_tuples(qc), qc.num_qubits); k = x["T_run_s"]
            else:
                _, k = cpu_bench.run_numpy(to_tuples(qc), qc.num_qubits)
            return time.perf_counter() - t0, k
    else:
        sys.exit("unknown platform")
    out = []
    for fam in a.families.split(","):
        for n in [int(x) for x in a.ns.split(",")]:
            qc = make(fam, n)
            e2e, ker = [], []
            for i in range(a.runs + 20):
                t, k = run(qc); e2e.append(t); ker.append(k)
            out.append(dict(platform=a.platform, family=fam, n=n, warmup=20, T_e2e=e2e, T_kernel=ker))
            x = np.array(e2e[20:]) * 1e3
            print(f"{a.platform} {qc.name:12s} e2e ms p50={np.percentile(x,50):.3f} p99={np.percentile(x,99):.3f} "
                  f"max={x.max():.3f} (p99-p50)/p50={(np.percentile(x,99)-np.percentile(x,50))/np.percentile(x,50):.3f}", flush=True)
            json.dump(dict(args=vars(a), records=out), open(a.out, "w"))


if __name__ == "__main__":
    main()
