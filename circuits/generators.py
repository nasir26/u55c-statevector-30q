"""Benchmark circuit generators. Same instances on every platform; seeds fixed.

Families: ghz, qft, rqc (depth d), vqe (hardware-efficient ansatz, L layers), grover.
`to_tuples` converts a Qiskit circuit to the (name, qubits, params) tuples used by the FPGA hosts.
"""
import math
import numpy as np
from qiskit import QuantumCircuit, qasm2

SEED_BASE = 20261005


def ghz(n):
    qc = QuantumCircuit(n, name=f"ghz_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def qft(n, swaps=True):
    qc = QuantumCircuit(n, name=f"qft_{n}")
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(math.pi / 2 ** (j - k), k, j)
    if swaps:
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
    return qc


def rqc(n, depth=10, seed=None):
    """Brick-wall random circuit: per layer a random 1q gate from {rx, ry, rz, t, h} on every
    qubit, then CZ on alternating pairs (even layers start at 0, odd at 1)."""
    rng = np.random.default_rng(SEED_BASE + 1000 * n + depth if seed is None else seed)
    qc = QuantumCircuit(n, name=f"rqc_{n}_d{depth}")
    for layer in range(depth):
        for q in range(n):
            g = rng.integers(5)
            th = float(rng.uniform(0, 2 * math.pi))
            if g == 0: qc.rx(th, q)
            elif g == 1: qc.ry(th, q)
            elif g == 2: qc.rz(th, q)
            elif g == 3: qc.t(q)
            else: qc.h(q)
        for q in range(layer % 2, n - 1, 2):
            qc.cz(q, q + 1)
    return qc


def vqe(n, layers=2, seed=None):
    """Hardware-efficient ansatz: [RY, RZ on all qubits; CX ladder] x layers + final RY, RZ."""
    rng = np.random.default_rng(SEED_BASE + 7 * n + layers if seed is None else seed)
    qc = QuantumCircuit(n, name=f"vqe_{n}_L{layers}")
    for _ in range(layers):
        for q in range(n):
            qc.ry(float(rng.uniform(0, 2 * math.pi)), q)
            qc.rz(float(rng.uniform(0, 2 * math.pi)), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(float(rng.uniform(0, 2 * math.pi)), q)
        qc.rz(float(rng.uniform(0, 2 * math.pi)), q)
    return qc


def grover(n, marked=None, iters=None):
    marked = (2 ** n - 1) if marked is None else marked
    iters = int(math.floor(math.pi / 4 * math.sqrt(2 ** n))) if iters is None else iters
    qc = QuantumCircuit(n, name=f"grover_{n}")
    qc.h(range(n))
    for _ in range(iters):
        for q in range(n):  # oracle: phase flip on |marked>
            if not (marked >> q) & 1: qc.x(q)
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        for q in range(n):
            if not (marked >> q) & 1: qc.x(q)
        qc.h(range(n)); qc.x(range(n))  # diffuser
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        qc.x(range(n)); qc.h(range(n))
    return qc


FAMILIES = {"ghz": ghz, "qft": qft, "rqc": rqc, "vqe": vqe, "grover": grover}


def make(family, n, **kw):
    return FAMILIES[family](n, **kw)


def to_tuples(qc):
    out = []
    for ins in qc.data:
        qs = [qc.find_bit(q).index for q in ins.qubits]
        out.append((ins.operation.name, qs, [float(p) for p in ins.operation.params] or None))
    return out


def export_qasm(qc, path):
    with open(path, "w") as f:
        f.write(qasm2.dumps(qc))
