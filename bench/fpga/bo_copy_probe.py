"""Is xrt::bo::copy between BOs in different HBM banks device-side (fast) on the U55C? Times copies."""
import uuid  # noqa: F401
import sys, time, json
import numpy as np, pyxrt
bdf, xcl, out = sys.argv[1:4]
dev = pyxrt.device(bdf)
k = pyxrt.kernel(dev, dev.load_xclbin(xcl), "quantum_fpga_kernel_complex128")
recs = []
for mb in (16, 64, 256):
    sz = mb << 20
    a = pyxrt.bo(dev, sz, pyxrt.bo.flags.normal, k.group_id(0))
    b = pyxrt.bo(dev, sz, pyxrt.bo.flags.normal, k.group_id(1))
    va = np.frombuffer(a.map(), dtype=np.uint32); va[:] = np.arange(va.size, dtype=np.uint32)
    a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, sz, 0)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); b.copy(a, sz, 0, 0); ts.append(time.perf_counter() - t0)
    b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, sz, 0)
    ok = bool(np.array_equal(np.frombuffer(b.map(), dtype=np.uint32), va))
    # reference: host round trip
    t0 = time.perf_counter()
    a.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, sz, 0)
    t1 = time.perf_counter()
    b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, sz, 0)
    t2 = time.perf_counter()
    r = dict(MB=mb, copy_s=ts, copy_gbps=sz / min(ts) / 1e9, correct=ok, d2h_gbps=sz / (t1 - t0) / 1e9, h2d_gbps=sz / (t2 - t1) / 1e9)
    recs.append(r); print(r, flush=True)
json.dump(recs, open(out, "w"), indent=1)
