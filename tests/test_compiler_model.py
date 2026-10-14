"""Validate compiler + golden model against Qiskit Statevector on families and a full gate zoo,
with small L/k so that remaps, high-qubit passes and multi-PC slices are all exercised."""
import os, sys, itertools
import numpy as np
WS = os.environ.get("WS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path[:0] = [f"{WS}/host/sv", f"{WS}/circuits"]
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from compiler import compile_circuit
from model import run_program
from generators import make
sys.path.insert(0, f"{WS}/tests")
from circuit_zoo import zoo

fails = 0; worst = 0; cases = 0
for n in (5, 7, 9):
    circs = [zoo(n, s) for s in range(4)] + [make(f, n) for f in ("ghz", "qft", "rqc", "vqe")]
    circs += [random_circuit(n, 8, max_operands=3, seed=s) for s in range(3)]
    for qc in circs:
        qc = qc.remove_final_measurements(inplace=False) if qc.num_clbits else qc
        ref = Statevector(qc).data
        for P, k, fuse, tiling in itertools.product((1, 2, 4), (3, 4, 5), (True, False), (True, False)):
            L = n - int(np.log2(P))
            if k > L: continue
            for init in (None, list(range(n))[::-1]):
                prog, fmap, st = compile_circuit(qc, L, k, fuse_1q=fuse, tiling=tiling, initial_map=init)
                out = run_program(prog, n, P, fmap)
                inf = 1 - abs(np.vdot(ref, out)) ** 2
                cases += 1; worst = max(worst, inf)
                if inf > 1e-5:
                    fails += 1
                    if fails < 6: print("FAIL", qc.name, n, P, k, fuse, tiling, init is not None, inf, st)
print(f"cases={cases} fails={fails} worst_infidelity={worst:.2e}")
sys.exit(1 if fails else 0)
