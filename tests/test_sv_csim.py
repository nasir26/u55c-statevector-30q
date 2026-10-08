"""C-simulation of kernels/sv/sv_engine.cpp vs the numpy golden model, segment by segment.
Each PC slice is run through the C model separately (as each CU does on hardware)."""
import os, sys, subprocess, itertools
import numpy as np
WS = os.environ["WS"]
sys.path[:0] = [f"{WS}/host/sv", f"{WS}/circuits", f"{WS}/tests"]
from qiskit.quantum_info import Statevector
from compiler import compile_circuit
from model import run_segment, remap, unpermute
from generators import make
from circuit_zoo import zoo

TB = f"{WS}/tmp/sv_tb"; T = f"{WS}/tmp/csim"
os.makedirs(T, exist_ok=True)

def run_c(seg, state, P, L):
    seg.astype(np.int32).tofile(f"{T}/ins.bin")
    for p in range(P):
        sl = state[p << L:(p + 1) << L]
        sl.astype(np.complex64).tofile(f"{T}/st.bin")
        subprocess.run([TB, f"{T}/ins.bin", f"{T}/st.bin", f"{T}/out.bin"], check=True)
        sl[:] = np.fromfile(f"{T}/out.bin", dtype=np.complex64)

fails = 0; cases = 0; worst_seg = 0; worst_f = 0
for n, P, k in [(7, 1, 5), (8, 2, 6), (9, 1, 7), (10, 4, 6), (12, 2, 9), (13, 1, 12), (14, 4, 12)]:
    L = n - int(np.log2(P))
    circs = [zoo(n, 1), make("qft", n), make("rqc", n), make("ghz", n), make("vqe", n)]
    for qc in circs:
        prog, fmap, st = compile_circuit(qc, L, k)
        sm = np.zeros(1 << n, np.complex64); sm[0] = 1; sc = sm.copy()
        for kind, payload in prog:
            if kind == "segment":
                run_segment(sm, P, payload); run_c(payload, sc, P, L)
                worst_seg = max(worst_seg, float(np.max(np.abs(sm - sc))))
            else:
                sm = remap(sm, n, payload); sc = remap(sc, n, payload)
        out = unpermute(sc, n, fmap)
        inf = 1 - abs(np.vdot(Statevector(qc).data, out)) ** 2
        cases += 1; worst_f = max(worst_f, inf)
        ok = inf < 1e-5
        fails += not ok
        print(f"n={n} P={P} k={k} {qc.name:14s} passes={st.n_passes} remaps={st.n_remaps} 1-F={inf:.1e} {'OK' if ok else 'FAIL'}", flush=True)
print(f"cases={cases} fails={fails} max|C-model| per segment={worst_seg:.2e} worst 1-F={worst_f:.2e}")
