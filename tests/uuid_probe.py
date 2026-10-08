"""Standalone reproducer: importing `uuid` after a pyxrt device is opened makes later
v07 kernel launches no-ops (state unchanged, run reports COMPLETED).
usage: uuid_probe.py <none|uuid|qiskit|uuid_first> <bdf> <xclbin>"""
import sys
mode, bdf, xcl = sys.argv[1:4]
if mode == 'uuid_first':
    import uuid  # noqa
import numpy as np, pyxrt
dev = pyxrt.device(bdf); k = pyxrt.kernel(dev, dev.load_xclbin(xcl), "quantum_fpga_kernel_complex128")
nq, nb = 6, 8; apb = (1 << nq) // nb
banks = [pyxrt.bo(dev, apb * 16, pyxrt.bo.flags.normal, k.group_id(i)) for i in range(nb)]
g = pyxrt.bo(dev, 4096, pyxrt.bo.flags.normal, k.group_id(8))
w = np.zeros(8, np.int32); w[0] = 2; w[1] = 0; w[2] = -1   # X on qubit 0
np.frombuffer(g.map(), np.int32, count=8)[:] = w; g.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 32, 0)
res = []
for i in range(6):
    for b in range(nb):
        v = np.frombuffer(banks[b].map(), np.float64, count=apb * 2); v[:] = 0
        if b == 0: v[0] = 1.0                                # |0>
        banks[b].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, apb * 16, 0)
    st = k(*banks, g, 1, nq).wait()
    banks[1].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, apb * 16, 0)
    ok = np.frombuffer(banks[1].map(), np.float64, count=2)[0] == 1.0   # expect |1> (bank 1, slot 0)
    res.append('ok' if ok else 'NOOP')
    if i == 0 and mode == 'uuid': import uuid  # noqa
    if i == 0 and mode == 'qiskit': import qiskit  # noqa
print(mode, res)
