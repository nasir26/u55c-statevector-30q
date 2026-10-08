"""PCIe H2D / D2H bandwidth on one card (pyxrt bo.sync), single BO and 8 BOs in parallel threads."""
import uuid  # noqa: F401
import sys, time, json, threading
import numpy as np, pyxrt
bdf, xcl, kname, out = sys.argv[1:5]
dev = pyxrt.device(bdf)
k = pyxrt.kernel(dev, dev.load_xclbin(xcl), kname)
TO, FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
recs = []
for mb in (1, 16, 128, 512):
    sz = mb << 20
    bos = [pyxrt.bo(dev, sz, pyxrt.bo.flags.normal, k.group_id(i)) for i in range(8)]
    for b in bos:
        np.frombuffer(b.map(), dtype=np.uint8)[:] = 1
    for dname, d in (("h2d", TO), ("d2h", FROM)):
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); bos[0].sync(d, sz, 0); ts.append(time.perf_counter() - t0)
        single = sz / min(ts) / 1e9
        tsp = []
        for _ in range(3):
            th = [threading.Thread(target=b.sync, args=(d, sz, 0)) for b in bos]
            t0 = time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]; tsp.append(time.perf_counter() - t0)
        par = 8 * sz / min(tsp) / 1e9
        r = dict(bdf=bdf, MB=mb, dir=dname, single_gbps=single, single_ts=ts, par8_gbps=par, par8_ts=tsp)
        recs.append(r); print(f"{mb:4d} MB {dname} single {single:6.2f} GB/s  8 threads {par:6.2f} GB/s", flush=True)
json.dump(recs, open(out, "w"), indent=1)
