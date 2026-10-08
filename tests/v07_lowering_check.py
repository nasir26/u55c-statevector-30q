"""Exact check of every v07 host lowering (fpga_simulator_v07._lower_to_kernel_basis) against Qiskit's
unitary for the gate, both qubit orders, using Qiskit Operator (no hardware). Global phase is ignored
(process fidelity |Tr(U^dag V)|/d)."""
import os, sys, math, json
import numpy as np
WS = os.environ["WS"]; sys.path.insert(0, f"{WS}/host/v07")
import fpga_simulator_v07 as v07
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def build(n, tuples):
    qc = QuantumCircuit(n)
    for nm, qs, ps in tuples:
        nm = {"cnot": "cx", "phase": "p", "u1": "p"}.get(nm, nm)
        if nm == "u2": qc.u(math.pi / 2, ps[0], ps[1], *qs)
        elif nm == "u3": qc.u(*ps, *qs)
        else: getattr(qc, nm)(*(list(ps or []) + list(qs)))
    return Operator(qc).data

th = 0.73
cases = [("cy", None), ("ch", None), ("swap", None), ("cp", [th]), ("crz", [th]), ("crx", [th]), ("cry", [th]),
         ("csx", None), ("dcx", None), ("iswap", None), ("ecr", None), ("rxx", [th]), ("ryy", [th]), ("rzz", [th]),
         ("rzz", [math.pi / 2]), ("ccx", None)]
out = []
for nm, ps in cases:
    for qs in ([0, 1], [1, 0]) if nm != "ccx" else ([0, 1, 2], [2, 0, 1]):
        n = len(qs)
        ref = build(n, [(nm, qs, ps)])
        low = v07._lower_to_kernel_basis([(nm, qs, ps)])
        got = build(n, low)
        pf = abs(np.trace(ref.conj().T @ got)) / ref.shape[0]
        out.append(dict(gate=nm, qubits=qs, params=ps, n_lowered=len(low), process_fidelity=float(pf ** 2)))
        print(f"{nm:6s} {str(qs):10s} lowered->{len(low):2d} gates  process fidelity {pf**2:.6f} {'OK' if pf**2 > 1 - 1e-9 else 'WRONG'}")
json.dump(out, open(f"{WS}/results/raw/phase1/v07_lowering_check.json", "w"), indent=1)
