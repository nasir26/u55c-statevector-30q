"""Host driver for kernels/hbm_stream (HBM ceiling microbenchmark).

Achieved bandwidth = bytes moved / T_kernel, where T_kernel is XRT launch -> wait (host clock)
and bytes = active_PCs * n_words * 64 B * reps * (2 for copy modes, 1 otherwise).
The kernel also returns its per-PC loop-iteration count, used to check the work was done.
"""
import uuid  # noqa: F401  must precede pyxrt device init (docs/BASELINE_AUDIT.md A)
import argparse, json, os, sys, time
import numpy as np
import pyxrt

MODES = {0: "read", 1: "write", 2: "copy_nonburst", 3: "copy_burst"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdf", required=True)
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--nwords", type=int, default=1 << 20)       # 64 MiB per PC per rep
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--sweep", default="pcs,pattern")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.bdf[5:7] in ("01", "21"):
        sys.exit("cards 01/21 not allowed")
    dev = pyxrt.device(a.bdf)
    k = pyxrt.kernel(dev, dev.load_xclbin(a.xclbin), "hbm_stream")
    n = a.nwords
    bo_bytes = (4 * n + 2) * 64
    bos = [pyxrt.bo(dev, bo_bytes, pyxrt.bo.flags.normal, k.group_id(i)) for i in range(32)]
    for b in bos:  # deterministic contents
        np.frombuffer(b.map(), dtype=np.uint64)[:] = 0x0123456789ABCDEF
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, bo_bytes, 0)

    def run(mode, npc, blk, skip, reps, spread=False):
        if spread:
            step = 32 // npc
            mask = sum(1 << (i * step) for i in range(npc))
        else:
            mask = (1 << npc) - 1 if npc < 32 else 0xFFFFFFFF
        # words actually touched per PC: blocks of blk out of each (blk+skip), over n words
        nblk = (n + blk - 1) // blk
        words = nblk * blk
        ts = []
        for _ in range(a.trials):
            t0 = time.perf_counter_ns()
            k(*bos, mode, n, blk, skip, reps, mask).wait()
            ts.append((time.perf_counter_ns() - t0) / 1e9)
        # verify the iteration counter on one active PC
        p0 = (mask & -mask).bit_length() - 1
        bos[p0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, 128, 4 * n * 64)
        it = int(np.frombuffer(bos[p0].map(), dtype=np.uint64, count=(4 * n + 2) * 8)[(4 * n + 1) * 8])
        mult = 2 if mode in (2, 3) else 1
        byts = npc * words * 64 * reps * mult
        med = float(np.median(ts))
        return dict(mode=MODES[mode], pcs=npc, mask=hex(mask), spread=spread, blk=blk, skip=skip,
                    n_words=n, reps=reps, bytes=byts, t_s=ts, t_med_s=med, gbps_med=byts / med / 1e9,
                    gbps_best=byts / min(ts) / 1e9, iter_count=it, iter_expected=words * reps)

    recs = []
    if "pcs" in a.sweep:
        for mode in (0, 1, 2, 3):
            for npc in (1, 2, 4, 8, 16, 32):
                for spread in ((False, True) if npc in (4, 8, 16) else (False,)):
                    r = run(mode, npc, n, 0, a.reps, spread)
                    recs.append(r)
                    print(f"{r['mode']:14s} pcs={npc:2d} spread={spread!s:5s} {r['gbps_med']:7.2f} GB/s "
                          f"it_ok={r['iter_count'] == r['iter_expected']}", flush=True)
    if "pattern" in a.sweep:
        # strided half-access, as a gate on qubit q touches 2^q-amplitude blocks (complex64: 8 amps/word)
        for mode in (0, 2):
            for blk in (1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096):
                r = run(mode, 32, blk, blk, a.reps)
                recs.append(r)
                print(f"{r['mode']:14s} pcs=32 blk={blk:5d} skip={blk:5d} {r['gbps_med']:7.2f} GB/s "
                      f"it_ok={r['iter_count'] == r['iter_expected']}", flush=True)
    with open(a.out, "w") as f:
        json.dump(dict(bdf=a.bdf, xclbin=a.xclbin, records=recs), f, indent=1)


if __name__ == "__main__":
    main()
