"""Benchmark the new SV kernel builds on the benchmark circuits (same generators/seeds as all platforms).

Each record carries the revision Sec. 6 timing split from host/sv/runtime.py, the compile statistics
(passes, remaps, fused gates), fidelity vs Qiskit (n <= --fid-max-n), and card power from the XMC sampler.
Ablation knobs: --p-max (active PCs/CUs), --k (tile log2 size), --fuse, --tiling, --layout (block | modulo:
initial map puts the LOW logical qubits on bank positions), --no-device-swap (host-mediated remaps).
--energy-loop S repeats each circuit for >= S seconds to get steady-state power (1 Hz sensors).
"""
import uuid  # noqa: F401  must precede pyxrt device init
import argparse, json, os, sys, time, platform
import numpy as np

WS = os.environ.get("WS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path[:0] = [f"{WS}/host/sv", f"{WS}/circuits", f"{WS}/bench/power"]
from runtime import SVRuntime  # noqa: E402
from generators import make  # noqa: E402
from fpga_power import Sampler, wait_cool  # noqa: E402


def initial_map(layout, n, P):
    if layout == "block":
        return None
    b = P.bit_length() - 1
    if layout == "modulo":      # logical qubit l -> physical: low b logical qubits on the bank bits
        return [(n - b + l) if l < b else (l - b) for l in range(n)]
    raise ValueError(layout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdf", required=True)
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--engines", type=int, required=True)
    ap.add_argument("--swaps", type=int, default=0)
    ap.add_argument("--families", default="ghz,qft,rqc,vqe")
    ap.add_argument("--ns", default="4,8,12,16,20,24")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--p-max", type=int, default=None)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--fuse", type=int, default=1)
    ap.add_argument("--tiling", type=int, default=1)
    ap.add_argument("--layout", default="block")
    ap.add_argument("--no-device-swap", action="store_true")
    ap.add_argument("--fid-max-n", type=int, default=24)
    ap.add_argument("--energy-loop", type=float, default=0.0)
    ap.add_argument("--rqc-depths", default="10")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cool-c", type=int, default=int(os.environ.get("SV_COOL_C", 80)),
                    help="wait until FPGA temperature < this before each run (env SV_COOL_C)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rt = SVRuntime(a.bdf, a.xclbin, a.engines, a.swaps, use_device_swap=not a.no_device_swap)
    pw = Sampler([a.bdf], a.out + ".power.csv").start()
    from qiskit.quantum_info import Statevector
    recs = []
    for fam in a.families.split(","):
        for n in [int(x) for x in a.ns.split(",")]:
            for depth in ([int(d) for d in a.rqc_depths.split(",")] if fam == "rqc" else [None]):
                qc = make(fam, n, depth=depth) if depth else make(fam, n)
                P, n_run, L = rt.plan(n, a.p_max)
                imap = initial_map(a.layout, n_run, P)
                ref = None
                for r in range(a.reps + 1):                    # rep -1 = warm-up
                    if os.path.exists(pw.abort_flag):
                        sys.exit("thermal abort flag set: " + open(pw.abort_flag).read())
                    cooled_s = wait_cool(a.bdf, a.cool_c)      # pacing: start every run below cool_c
                    t0 = time.perf_counter()
                    res = rt.run(qc, k=a.k, p_max=a.p_max, fuse=bool(a.fuse), tiling=bool(a.tiling),
                                 initial_map=imap, return_state=n <= a.fid_max_n, seed=r)
                    t1 = time.perf_counter()
                    fid = None
                    if n <= a.fid_max_n:
                        if ref is None:
                            ref = Statevector(qc).data
                        sv = res["state"].astype(np.complex128)   # float64 accumulation (complex64 vdot overshoots F)
                        fid = float(abs(np.vdot(ref, sv)) ** 2 / np.vdot(sv, sv).real)
                    rec = dict(platform="fpga_sv", tag=a.tag, xclbin=os.path.basename(os.path.dirname(a.xclbin)),
                               engines=a.engines, swaps=a.swaps, family=fam, n=n, depth=depth, rep=r - 1,
                               warmup=(r == 0), P=res["P"], L=res["L"], k=res["k"], n_run=res["n_run"],
                               fuse=a.fuse, tiling=a.tiling, layout=a.layout, device_swap=not a.no_device_swap,
                               fidelity=fid, wall_s=t1 - t0, stats=res["stats"], t0=t0, t1=t1, cooled_wait_s=cooled_s,
                               **res["timing"])
                    recs.append(rec)
                    T = res["timing"]
                    print(f"{qc.name:16s} rep{r - 1:2d} P={res['P']:2d} kernel {T['T_kernel']:9.4f}s exch {T['T_exchange']:8.4f}s "
                          f"e2e {T['T_e2e']:9.4f}s passes={res['stats']['n_passes']} remaps={res['stats']['n_remaps']} F={fid}",
                          flush=True)
                if a.energy_loop > 0:
                    t0 = time.perf_counter(); cnt = 0; tk = 0.0
                    while time.perf_counter() - t0 < a.energy_loop:
                        res = rt.run(qc, k=a.k, p_max=a.p_max, fuse=bool(a.fuse), tiling=bool(a.tiling),
                                     initial_map=imap, return_state=False)
                        cnt += 1; tk += res["timing"]["T_kernel"]
                    t1 = time.perf_counter()
                    recs.append(dict(platform="fpga_sv", tag=a.tag, family=fam, n=n, depth=depth, energy_loop=True, runs=cnt,
                                     loop_s=t1 - t0, per_run_s=(t1 - t0) / cnt, kernel_frac=tk / (t1 - t0), t0=t0, t1=t1))
                with open(a.out, "w") as f:
                    json.dump(dict(host=platform.node(), bdf=a.bdf, xclbin=a.xclbin, args=vars(a), T_setup_s=rt.T_setup,
                                   records=recs), f, indent=1)
    pw.stop()
    # attach mean board power for every record window (sampler is 1 Hz; short windows get the enclosing reading)
    for rec in recs:
        rec["board_w"] = pw.mean_power(a.bdf, rec["t0"], rec["t1"])
        rec["power_readings"] = pw.n_readings(a.bdf, rec["t0"], rec["t1"])
    with open(a.out, "w") as f:
        json.dump(dict(host=platform.node(), bdf=a.bdf, xclbin=a.xclbin, args=vars(a), T_setup_s=rt.T_setup,
                       records=recs), f, indent=1)


if __name__ == "__main__":
    main()
