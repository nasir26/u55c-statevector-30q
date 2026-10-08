"""Steady-state board power and energy per circuit for the v07 baseline: repeat each circuit for >= --loop s
with the XMC sampler running (1 Hz sensors), also record idle power before the loops."""
import uuid  # noqa: F401
import argparse, json, os, sys, time
WS = os.environ["WS"]; sys.path[:0] = [f"{WS}/host/v07", f"{WS}/circuits", f"{WS}/bench/power"]
import fpga_simulator_v07 as v07
from generators import make, to_tuples
from fpga_power import Sampler
ap = argparse.ArgumentParser()
ap.add_argument("--bdf", required=True); ap.add_argument("--xclbin", required=True)
ap.add_argument("--ns", default="20,24"); ap.add_argument("--families", default="ghz,rqc,vqe")
ap.add_argument("--loop", type=float, default=30); ap.add_argument("--out", required=True)
a = ap.parse_args()
eng = v07.QuantumFPGACircuit(a.xclbin, device_index=a.bdf)
pw = Sampler([a.bdf], a.out + ".power.csv").start()
t0 = time.perf_counter(); time.sleep(10); idle = pw.mean_power(a.bdf, t0, time.perf_counter())
recs = [dict(kind="idle_loaded_xclbin", board_w=idle, window_s=10)]
for fam in a.families.split(","):
    for n in [int(x) for x in a.ns.split(",")]:
        g = to_tuples(make(fam, n)); t0 = time.perf_counter(); cnt = 0; tk = 0.0
        while time.perf_counter() - t0 < a.loop:
            eng.run_circuit(n, g, shots=1024, validate=False); cnt += 1; tk += eng._t_launch_us / 1e6
        t1 = time.perf_counter()
        p = pw.mean_power(a.bdf, t0, t1)
        recs.append(dict(platform="fpga_v07", family=fam, n=n, runs=cnt, loop_s=t1 - t0, per_run_s=(t1 - t0) / cnt,
                         kernel_frac=tk / (t1 - t0), board_w=p, energy_per_run_j=p * (t1 - t0) / cnt,
                         readings=pw.n_readings(a.bdf, t0, t1)))
        print(recs[-1], flush=True)
pw.stop()
json.dump(dict(args=vars(a), records=recs), open(a.out, "w"), indent=1)
