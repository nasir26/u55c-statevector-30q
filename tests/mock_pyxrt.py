"""Minimal pyxrt stand-in for testing host orchestration without hardware.

BOs are numpy byte buffers. 'sv_engine' CUs execute the instruction segment stored at word offset ins_off of their
BO using the numpy golden model (validated against the HLS C model); 'sv_swap' CUs swap two word ranges.
Each BO records which CU 'owns' it, and an engine refuses a BO from another CU (mirrors the PC connectivity rule).
"""
import sys, types
import numpy as np


class _Dir:
    XCL_BO_SYNC_BO_TO_DEVICE = 0
    XCL_BO_SYNC_BO_FROM_DEVICE = 1


class bo:
    class flags:
        normal = 0

    def __init__(self, dev, size, flags, group):
        self.buf = np.zeros(size, dtype=np.uint8)
        self.group = group

    def map(self):
        return memoryview(self.buf)

    def sync(self, d, size, off):
        pass


class _Run:
    def wait(self, *a):
        return "COMPLETED"


class kernel:
    def __init__(self, dev, uuid, name):
        self.name = name
        self.kind = name.split(":")[0]

    def group_id(self, i):
        return self.name

    def __call__(self, *args):
        if self.kind == "sv_engine":
            st, ins_off = args
            assert st.group in (self.name, getattr(self, "bank", None)) or isinstance(st.group, int), \
                f"BO of {st.group} passed to {self.name} (PC connectivity violation)"
            from model import run_segment
            seg = st.buf[ins_off * 32:].view(np.int32)
            L = int(seg[1])
            sl = st.buf[: (1 << L) * 8].view(np.complex64)
            run_segment(sl, 1, seg)
        elif self.kind == "sv_swap":
            a, b, offa, offb, nw = args
            A = a.buf[offa * 64:(offa + nw) * 64]; B = b.buf[offb * 64:(offb + nw) * 64]
            t = A.copy(); A[:] = B; B[:] = t
        return _Run()


class device:
    def __init__(self, bdf):
        self.bdf = bdf

    def load_xclbin(self, path):
        return "uuid"


def install():
    m = types.ModuleType("pyxrt")
    m.device, m.kernel, m.bo, m.xclBOSyncDirection = device, kernel, bo, _Dir
    sys.modules["pyxrt"] = m
