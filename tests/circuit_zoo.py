"""Random circuit over the full Qiskit standard gate zoo (shared by tests)."""
import numpy as np
from qiskit import QuantumCircuit


def zoo(n, seed):
    rng = np.random.default_rng(seed); qc = QuantumCircuit(n)
    one = ["h","x","y","z","s","sdg","t","tdg","sx","sxdg"]
    par1 = ["rx","ry","rz","p"]
    two = ["cx","cy","cz","ch","swap","iswap","ecr","dcx","csx"]
    par2 = ["rxx","ryy","rzz","rzx","cp","crx","cry","crz"]
    for _ in range(12 * n):
        r = rng.integers(6)
        q = [int(x) for x in rng.choice(n, 3, replace=False)]
        th = float(rng.uniform(-np.pi, np.pi))
        if r == 0: getattr(qc, rng.choice(one))(q[0])
        elif r == 1: getattr(qc, rng.choice(par1))(th, q[0])
        elif r == 2: getattr(qc, rng.choice(two))(q[0], q[1])
        elif r == 3: getattr(qc, rng.choice(par2))(th, q[0], q[1])
        elif r == 4: qc.u(th, th / 2, -th / 3, q[0])
        else: qc.ccx(q[0], q[1], q[2])
    return qc
