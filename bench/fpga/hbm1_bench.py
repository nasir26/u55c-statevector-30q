"""Host driver for the multi-CU HBM ceiling microbenchmark (kernels/hbm_stream/hbm1.cpp).

Achieved bandwidth = bytes moved by all active CUs / wall time from starting all CU runs to the last completion.
bytes = active * n_words * word_bytes * reps * (2 for copy, 1 otherwise). Each point: median of --trials.
"""
import uuid  # noqa: F401
import argparse, json, sys, time
import numpy as np
import pyxrt

MODES = {0: "read", 1: "write", 3: "copy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdf", required=True)
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--cus", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--mb", type=int, default=128, help="MiB per CU per repetition")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.bdf[5:7] in ("01", "21"):
        sys.exit("cards 01/21 not allowed")
    dev = pyxrt.device(a.bdf)
    uid = dev.load_xclbin(a.xclbin)
    ks = [pyxrt.kernel(dev, uid, f"hbm1:{{hbm1_{i + 1}}}") for i in range(a.cus)]
    wb = a.width // 8
    n = (a.mb << 20) // wb
    bos = []
    for k in ks:
        b = pyxrt.bo(dev, (2 * n + 1) * wb, pyxrt.bo.flags.normal, k.group_id(0))
        np.frombuffer(b.map(), dtype=np.uint8)[:] = 7
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, (2 * n + 1) * wb, 0)
        bos.append(b)
    recs = []
    counts = sorted({1, 2, 4, 8, 16, a.cus} & set(range(1, a.cus + 1)))
    for mode in (0, 1, 3):
        for c in counts:
            for spread in ((False, True) if 1 < c < a.cus else (False,)):
                idx = list(range(0, a.cus, a.cus // c))[:c] if spread else list(range(c))
                ts = []
                for _ in range(a.trials):
                    t0 = time.perf_counter()
                    runs = [ks[i](bos[i], mode, n, a.reps) for i in idx]
                    for r in runs:
                        r.wait()
                    ts.append(time.perf_counter() - t0)
                byts = c * n * wb * a.reps * (2 if mode == 3 else 1)
                med = float(np.median(ts))
                r = dict(mode=MODES[mode], cus=c, spread=spread, cu_idx=idx, width=a.width, mb_per_cu=a.mb, reps=a.reps,
                         bytes=byts, t_s=ts, gbps_med=byts / med / 1e9, gbps_best=byts / min(ts) / 1e9,
                         gbps_per_cu=byts / med / 1e9 / c)
                recs.append(r)
                print(f"{r['mode']:5s} CUs={c:2d} spread={spread!s:5s} {r['gbps_med']:7.1f} GB/s ({r['gbps_per_cu']:5.2f}/CU)", flush=True)
    json.dump(dict(bdf=a.bdf, xclbin=a.xclbin, cus=a.cus, width=a.width, records=recs), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
