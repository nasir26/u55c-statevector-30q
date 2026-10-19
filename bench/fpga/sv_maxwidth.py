"""Largest-width verification on one card (revision, reviewer item R5.7e): run each benchmark family at
n = 26 to 30 on the flagship build, record the timing split and save the final logical-order state to disk.
The card is parked on the given bitstream when the script ends (also on error), so that the CPU reference
computation of bench/fpga/sv_maxwidth_ref.py runs with the card cool and the engine unloaded.
"""
import uuid  # noqa: F401  must precede pyxrt device init
import argparse, json, os, sys, time, platform
import numpy as np

WS = os.environ.get("WS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path[:0] = [f"{WS}/host/sv", f"{WS}/circuits", f"{WS}/bench/power"]
from runtime import SVRuntime  # noqa: E402
from generators import make  # noqa: E402
from fpga_power import Sampler, wait_cool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdf", required=True)
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--park-xclbin", required=True)
    ap.add_argument("--engines", type=int, default=16)
    ap.add_argument("--swaps", type=int, default=4)
    ap.add_argument("--families", default="ghz,qft,rqc,vqe")
    ap.add_argument("--ns", default="26,28,29,30")
    ap.add_argument("--rqc-depth", type=int, default=10)
    ap.add_argument("--cool-c", type=int, default=80)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.state_dir, exist_ok=True)
    recs = []
    pw = Sampler([a.bdf], a.out + ".power.csv").start()
    rt = None
    try:
        rt = SVRuntime(a.bdf, a.xclbin, a.engines, a.swaps)
        for fam in a.families.split(","):
            for n in [int(x) for x in a.ns.split(",")]:
                qc = make(fam, n, depth=a.rqc_depth) if fam == "rqc" else make(fam, n)
                for rep in (-1, 0):                       # warm-up without state return, then the verified run
                    if os.path.exists(pw.abort_flag):
                        raise SystemExit("thermal abort flag set: " + open(pw.abort_flag).read())
                    cooled = wait_cool(a.bdf, a.cool_c)
                    t0 = time.perf_counter()
                    res = rt.run(qc, return_state=(rep == 0), seed=rep + 1)
                    t1 = time.perf_counter()
                    rec = dict(platform="fpga_sv", tag="maxwidth", xclbin=os.path.basename(os.path.dirname(a.xclbin)),
                               family=fam, n=n, depth=a.rqc_depth if fam == "rqc" else None, rep=rep, warmup=(rep < 0),
                               P=res["P"], L=res["L"], k=res["k"], stats=res["stats"], t0=t0, t1=t1, wall_s=t1 - t0,
                               cooled_wait_s=cooled, **res["timing"])
                    if rep == 0:
                        p = os.path.join(a.state_dir, f"{fam}_{n}.npy")
                        np.save(p, res["state"])
                        rec["state_file"] = os.path.basename(p)
                        del res["state"]
                    recs.append(rec)
                    T = res["timing"]
                    print(f"{qc.name:16s} rep{rep:2d} P={res['P']:2d} kernel {T['T_kernel']:8.3f}s exch {T['T_exchange']:7.3f}s "
                          f"e2e {T['T_e2e']:8.3f}s passes={res['stats']['n_passes']} remaps={res['stats']['n_remaps']}", flush=True)
                json.dump(dict(host=platform.node(), bdf=a.bdf, xclbin=a.xclbin, args=vars(a), records=recs),
                          open(a.out, "w"), indent=1)
    finally:
        pw.stop()
        for rec in recs:
            rec["board_w"] = pw.mean_power(a.bdf, rec["t0"], rec["t1"])
        json.dump(dict(host=platform.node(), bdf=a.bdf, xclbin=a.xclbin, args=vars(a), records=recs),
                  open(a.out, "w"), indent=1)
        del rt
        import pyxrt
        pyxrt.device(a.bdf).load_xclbin(a.park_xclbin)
        print("parked", a.bdf, "on", os.path.basename(a.park_xclbin), flush=True)


if __name__ == "__main__":
    main()
