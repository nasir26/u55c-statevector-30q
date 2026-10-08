"""Direct driver for the v07 kernel (quantum_fpga_kernel_complex128, 8-bank).

Bypasses the v07 host lowering so each native opcode is tested in isolation.
Layout (from kernels/v07/quantum_fpga_kernel_unified.cpp): amplitude i lives in
bank i & 7 at slot i >> 3, stored as [re, im] float64.

Every bank BO is allocated with a guard region after the state slice; the guard
is filled with a sentinel and checked after each launch to detect
out-of-bounds writes.
"""
import argparse, json, os, sys, time
# Must precede pyxrt device init: importing `uuid` (and hence qiskit) after a
# device is opened makes later kernel launches report COMPLETED without
# executing (found 2026-10-05, see docs/BASELINE_AUDIT.md).
import uuid  # noqa: F401
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import pyxrt

NB = 8
GW = 8
OPS = {'id': 0, 'h': 1, 'x': 2, 'y': 3, 'z': 4, 's': 5, 't': 6, 'sdg': 7, 'tdg': 8,
       'sx': 9, 'sxdg': 10, 'rx': 11, 'ry': 12, 'rz': 13, 'p': 14, 'u2': 15, 'u3': 16,
       'cx': 17, 'cz': 18}
SENTINEL = 12345.678


class V07:
    def __init__(self, bdf, xclbin, max_nq, guard_factor=2):
        self.dev = pyxrt.device(bdf)
        self.uuid = self.dev.load_xclbin(xclbin)
        self.k = pyxrt.kernel(self.dev, self.uuid, "quantum_fpga_kernel_complex128")
        self.max_nq = max_nq
        apb = (1 << max_nq) // NB
        self.state_bytes = apb * 16
        self.alloc_bytes = self.state_bytes * guard_factor
        self.banks = [pyxrt.bo(self.dev, self.alloc_bytes, pyxrt.bo.flags.normal, self.k.group_id(i))
                      for i in range(NB)]
        self.gbo = pyxrt.bo(self.dev, 512 * GW * 4, pyxrt.bo.flags.normal, self.k.group_id(8))

    def _guard_pattern(self, b):
        # unique per word and per bank, so swaps between guard words are detected
        return SENTINEL + b * 1e7 + np.arange(self.alloc_bytes // 8, dtype=np.float64)

    def _view(self, b):
        return np.frombuffer(self.banks[b].map(), dtype=np.float64, count=self.alloc_bytes // 8)

    def upload(self, sv, nq):
        apb = (1 << nq) // NB
        for b in range(NB):
            v = self._view(b)
            v[:] = self._guard_pattern(b)
            v[0:2 * apb:2] = sv.real[b::NB]
            v[1:2 * apb:2] = sv.imag[b::NB]
            self.banks[b].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, self.alloc_bytes, 0)

    def download(self, nq):
        apb = (1 << nq) // NB
        sv = np.empty(1 << nq, dtype=np.complex128)
        guard_hits = 0
        for b in range(NB):
            self.banks[b].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, self.alloc_bytes, 0)
            v = self._view(b)
            sv.real[b::NB] = v[0:2 * apb:2]
            sv.imag[b::NB] = v[1:2 * apb:2]
            guard_hits += int(np.count_nonzero(v[2 * apb:] != self._guard_pattern(b)[2 * apb:]))
        return sv, guard_hits

    @staticmethod
    def encode(gates):
        enc = np.zeros((len(gates), GW), dtype=np.int32)
        for i, (nm, qs, ps) in enumerate(gates):
            enc[i, 0] = OPS[nm]
            enc[i, 1] = qs[0]
            enc[i, 2] = qs[1] if len(qs) > 1 else -1
            enc[i, 3] = -1
            pf = np.zeros(3, dtype=np.float32)
            if ps:
                pf[:len(ps)] = ps
            enc[i, 4:7] = pf.view(np.int32)
        return enc.reshape(-1)

    def run(self, gates, nq):
        arr = self.encode(gates)
        np.frombuffer(self.gbo.map(), dtype=np.int32, count=arr.size)[:] = arr
        self.gbo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, arr.nbytes, 0)
        t0 = time.perf_counter_ns()
        r = self.k(*self.banks, self.gbo, len(gates), nq)
        r.wait()
        return (time.perf_counter_ns() - t0) / 1e3


def reference(sv, nq, nm, qs, ps):
    qc = QuantumCircuit(nq)
    p = [float(np.float32(x)) for x in (ps or [])]  # kernel receives float32 angles
    {'id': lambda: qc.id(*qs), 'h': lambda: qc.h(*qs), 'x': lambda: qc.x(*qs), 'y': lambda: qc.y(*qs),
     'z': lambda: qc.z(*qs), 's': lambda: qc.s(*qs), 't': lambda: qc.t(*qs), 'sdg': lambda: qc.sdg(*qs),
     'tdg': lambda: qc.tdg(*qs), 'sx': lambda: qc.sx(*qs), 'sxdg': lambda: qc.sxdg(*qs),
     'rx': lambda: qc.rx(p[0], *qs), 'ry': lambda: qc.ry(p[0], *qs), 'rz': lambda: qc.rz(p[0], *qs),
     'p': lambda: qc.p(p[0], *qs), 'u2': lambda: qc.u(np.pi / 2, p[0], p[1], *qs),
     'u3': lambda: qc.u(p[0], p[1], p[2], *qs), 'cx': lambda: qc.cx(*qs), 'cz': lambda: qc.cz(*qs)}[nm]()
    return Statevector(sv).evolve(qc).data


def haar(nq, rng):
    v = rng.normal(size=1 << nq) + 1j * rng.normal(size=1 << nq)
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdf", required=True)
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--nq", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    if a.bdf.startswith("0000:01") or a.bdf.startswith("0000:21"):
        sys.exit("refusing to use cards 01/21 (author decision)")
    rng = np.random.default_rng(a.seed)
    dev = V07(a.bdf, a.xclbin, a.nq)
    nq = a.nq
    recs = []
    params = {'rx': [0.7], 'ry': [1.1], 'rz': [0.4], 'p': [0.9], 'u2': [0.3, 1.2], 'u3': [0.5, 0.8, 1.4]}
    cases = []
    for nm in OPS:
        if nm in ('cx', 'cz'):
            for c in range(nq):
                for t in range(nq):
                    if c != t:
                        cases.append((nm, [c, t], None))
        else:
            for q in range(nq):
                cases.append((nm, [q], params.get(nm)))
    for nm, qs, ps in cases:
        for kind in ('basis', 'haar'):
            if kind == 'basis':
                sv = np.zeros(1 << nq, dtype=np.complex128)
                sv[int(rng.integers(1 << nq))] = 1.0
            else:
                sv = haar(nq, rng)
            dev.upload(sv, nq)
            us = dev.run([(nm, qs, ps)], nq)
            out, gh = dev.download(nq)
            ref = reference(sv, nq, nm, qs, ps)
            fid = float(abs(np.vdot(ref, out)) ** 2 / (np.vdot(out, out).real * np.vdot(ref, ref).real))
            maxerr = float(np.max(np.abs(ref - out)))
            recs.append(dict(gate=nm, qubits=qs, params=ps, state=kind, fidelity=fid, infidelity=1 - fid,
                             max_abs_err=maxerr, norm=float(np.linalg.norm(out)), guard_writes=gh, run_us=us))
    with open(a.out, "w") as f:
        json.dump(dict(bdf=a.bdf, xclbin=a.xclbin, nq=nq, seed=a.seed, records=recs), f, indent=1)
    bad = [r for r in recs if r['max_abs_err'] > 1e-6 or r['guard_writes']]
    print(f"{len(recs)} cases, {len(bad)} failing (max_abs_err>1e-6 or guard writes)")
    from collections import Counter
    print("failures by gate:", dict(Counter(r['gate'] for r in bad)))
    print("guard writes by gate:", dict(Counter(r['gate'] for r in recs if r['guard_writes'])))


if __name__ == "__main__":
    main()
