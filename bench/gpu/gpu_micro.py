"""GPU device-level microbenchmarks (Phase 2 ceilings, R3.8 analogue on the GPU side).

1. Device memory bandwidth: CuPy device-to-device copy of a large buffer (read + write), CUDA-event timed.
2. cuStateVec single-gate sweep: apply a dense 2x2 (H) and a controlled-X on a complex64/complex128 statevector
   of n qubits, target qubit low/middle/high; per-gate time from CUDA events (device time only), giving the
   achieved bytes/s of one statevector sweep (2 * 2^n * bytes_per_amp read + write).
Output: 'RESULT ' JSON lines on stdout.
"""
import json, sys, time
import numpy as np
import cupy as cp
from cuquantum.bindings import custatevec as cusv


def ev_time(fn, reps):
    s, e = cp.cuda.Event(), cp.cuda.Event()
    fn(); cp.cuda.Device().synchronize()
    s.record()
    for _ in range(reps):
        fn()
    e.record(); e.synchronize()
    return cp.cuda.get_elapsed_time(s, e) / 1e3 / reps


def bw_test():
    for gb in (1, 4, 16):
        n = (gb << 30) // 4
        a = cp.ones(n, dtype=cp.float32); b = cp.empty_like(a)
        t = ev_time(lambda: b.__setitem__(slice(None), a), 10)
        print("RESULT " + json.dumps(dict(test="d2d_copy", GB=gb, s=t, rw_gbps=2 * a.nbytes / t / 1e9)), flush=True)
        del a, b; cp.get_default_memory_pool().free_all_blocks()


def gate_sweep(n, dtype):
    ctype = cp.complex64 if dtype == "c64" else cp.complex128
    sv_type = cusv.cudaDataType.CUDA_C_32F if dtype == "c64" else cusv.cudaDataType.CUDA_C_64F
    comp = cusv.ComputeType.COMPUTE_32F if dtype == "c64" else cusv.ComputeType.COMPUTE_64F
    sv = cp.zeros(1 << n, dtype=ctype); sv[0] = 1
    h = cusv.create()
    H = (np.array([[1, 1], [1, -1]]) / np.sqrt(2)).astype(np.complex64 if dtype == "c64" else np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=H.dtype)
    Hd, Xd = cp.asarray(H), cp.asarray(X)
    out = []
    for gate, mat, nctl in (("H", Hd, 0), ("CX", Xd, 1)):
        for tgt in sorted({0, n // 2, n - 1}):
            ctl = [(tgt + 1) % n] if nctl else []
            targets = np.array([tgt], dtype=np.int32); controls = np.array(ctl, dtype=np.int32)
            ws = cusv.apply_matrix_get_workspace_size(h, sv_type, n, mat.data.ptr, sv_type, cusv.MatrixLayout.ROW, 0,
                                                      1, len(ctl), comp)
            wsp = cp.empty(max(ws, 1), dtype=cp.uint8)

            def apply():
                cusv.apply_matrix(h, sv.data.ptr, sv_type, n, mat.data.ptr, sv_type, cusv.MatrixLayout.ROW, 0,
                                  targets.ctypes.data, 1, controls.ctypes.data if ctl else 0, 0, len(ctl), comp,
                                  wsp.data.ptr, ws)
            reps = 20 if n <= 28 else 5
            t = ev_time(apply, reps)
            byts = 2 * (1 << n) * sv.itemsize * (0.5 if nctl else 1.0)   # CX touches half the amplitudes
            rec = dict(test="custatevec_gate", n=n, dtype=dtype, gate=gate, target=tgt, controls=ctl, s=t,
                       eff_gbps=byts / t / 1e9)
            print("RESULT " + json.dumps(rec), flush=True)
    cusv.destroy(h)
    del sv; cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    print("RESULT " + json.dumps(dict(meta="gpu", name=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                                       cusv_version=cusv.get_version())), flush=True)
    bw_test()
    for dt in ("c64", "c128"):
        for n in (16, 20, 24, 26, 28, 30, 32) + ((33,) if dt == "c64" else ()):
            try:
                gate_sweep(n, dt)
            except Exception as e:
                print("RESULT " + json.dumps(dict(test="custatevec_gate", n=n, dtype=dt, error=str(e)[:200])), flush=True)
