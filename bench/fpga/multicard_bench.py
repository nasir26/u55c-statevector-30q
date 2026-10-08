"""Multi-card benchmark (preliminary, R4.5): the same circuits on 1 and C cards with host-mediated cross-card exchange.

Records the Sec. 6 timing split plus exchange_device_s / exchange_host_s, board power of every card, the card count
and slice layout. Each repetition starts only when every card is below --cool-c (thermal pacing).
"""
import uuid  # noqa: F401
import argparse, json, os, sys, time, platform
import numpy as np

WS = os.environ["WS"]
sys.path[:0] = [f"{WS}/host/sv", f"{WS}/circuits", f"{WS}/bench/power"]
from multicard import MultiSVRuntime  # noqa: E402
from generators import make  # noqa: E402
from fpga_power import Sampler, wait_cool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdfs", required=True, help="comma-separated card BDFs")
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--engines", type=int, default=16)
    ap.add_argument("--swaps", type=int, default=4)
    ap.add_argument("--families", default="ghz,qft,rqc,vqe")
    ap.add_argument("--ns", default="26,28,30,31")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cool-c", type=int, default=int(os.environ.get("SV_COOL_C", 80)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bdfs = a.bdfs.split(",")
    if any(b[5:7] in ("01", "21") for b in bdfs):
        sys.exit("cards 01/21 need explicit author approval")
    rt = MultiSVRuntime(bdfs, a.xclbin, a.engines, a.swaps)
    pw = Sampler(bdfs, a.out + ".power.csv").start()
    recs = []
    for fam in a.families.split(","):
        for n in [int(x) for x in a.ns.split(",")]:
            qc = make(fam, n)
            for r in range(a.reps + 1):
                if os.path.exists(pw.abort_flag):
                    sys.exit("thermal abort flag set: " + open(pw.abort_flag).read())
                waited = sum(wait_cool(b, a.cool_c) for b in bdfs)
                t0 = time.perf_counter()
                res = rt.run(qc, shots=1024, return_state=False, seed=r)
                t1 = time.perf_counter()
                rec = dict(platform="fpga_sv_multicard", cards=len(bdfs), bdfs=bdfs, family=fam, n=n, rep=r - 1,
                           warmup=(r == 0), P=res["P"], L=res["L"], stats=res["stats"], t0=t0, t1=t1,
                           cooled_wait_s=waited, **res["timing"])
                recs.append(rec)
                T = res["timing"]
                print(f"C={len(bdfs)} {qc.name:12s} rep{r - 1:2d} kernel {T['T_kernel']:8.3f} exch dev {T['exchange_device_s']:7.3f} "
                      f"host {T['exchange_host_s']:7.3f} e2e {T['T_e2e']:8.3f} passes={res['stats']['n_passes']} remaps={res['stats']['n_remaps']}",
                      flush=True)
            json.dump(dict(host=platform.node(), args=vars(a), records=recs), open(a.out, "w"), indent=1)
    pw.stop()
    for rec in recs:
        rec["board_w"] = {b: pw.mean_power(b, rec["t0"], rec["t1"]) for b in bdfs}
    json.dump(dict(host=platform.node(), args=vars(a), records=recs), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
