"""Fidelity-based functional suite, platform-agnostic.

Every case is a Qiskit circuit; the platform returns the final statevector, which is compared with Qiskit
Statevector (complex128). Generic input states: each gate test is preceded by a fixed random U3 layer plus a
CZ ring, so a wrong amplitude anywhere changes the fidelity (no 'norm only' tests).

Sections
  S1 1q gates      : 17 gates x every qubit position (n=5)
  S2 2q gates      : 21 gates x every ordered qubit pair (n=4)
  S3 3+q gates     : ccx, ccz, cswap, mcx (3,4 controls), both control orders
  S4 families      : ghz, qft, rqc(d10), vqe(L2), grover at n = 2..max_n_fid (step 2)
  S5 large-n       : analytic checks without a full reference (GHZ amplitudes, QFT|0> uniform) up to max_n
Pass: 1 - F <= thr (thr = 1e-5 for complex64 platforms, recorded in output).
"""
import os, sys, math, json, time
import numpy as np
WS = os.environ.get("WS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path[:0] = [f"{WS}/circuits"]
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from generators import make
from qiskit.circuit.library import XXPlusYYGate

ONE = [("h", 0), ("x", 0), ("y", 0), ("z", 0), ("s", 0), ("sdg", 0), ("t", 0), ("tdg", 0), ("sx", 0), ("sxdg", 0),
       ("rx", 1), ("ry", 1), ("rz", 1), ("p", 1), ("u", 3), ("r", 2), ("id", 0)]
TWO = [("cx", 0), ("cy", 0), ("cz", 0), ("ch", 0), ("swap", 0), ("iswap", 0), ("ecr", 0), ("dcx", 0), ("csx", 0),
       ("cp", 1), ("crx", 1), ("cry", 1), ("crz", 1), ("rxx", 1), ("ryy", 1), ("rzz", 1), ("rzx", 1), ("cu", 4),
       ("cs", 0), ("csdg", 0), ("xx_plus_yy", 2)]
PAR = [0.7311, -1.2207, 2.4113, 0.3359]


def prep(n, seed=7):
    rng = np.random.default_rng(seed); qc = QuantumCircuit(n)
    for q in range(n):
        qc.u(*rng.uniform(-math.pi, math.pi, 3), q)
    for q in range(n):
        qc.cz(q, (q + 1) % n) if n > 1 else None
    for q in range(n):
        qc.u(*rng.uniform(-math.pi, math.pi, 3), q)
    return qc


def cases(max_n_fid=20):
    out = []
    n = 5
    for g, np_ in ONE:
        for q in range(n):
            qc = prep(n); getattr(qc, g)(*PAR[:np_], q); out.append(("S1", f"{g}_q{q}", qc))
    n = 4
    for g, np_ in TWO:
        for a in range(n):
            for b in range(n):
                if a != b:
                    qc = prep(n)
                    if g == "xx_plus_yy":
                        qc.append(XXPlusYYGate(*PAR[:np_]), [a, b])
                    else:
                        getattr(qc, g)(*PAR[:np_], a, b)
                    out.append(("S2", f"{g}_{a}{b}", qc))
    n = 5
    for name, f in [("ccx", lambda qc: qc.ccx(0, 2, 4)), ("ccx_r", lambda qc: qc.ccx(4, 1, 0)),
                    ("ccz", lambda qc: qc.ccz(1, 3, 2)), ("cswap", lambda qc: qc.cswap(0, 3, 4)),
                    ("cswap_r", lambda qc: qc.cswap(4, 1, 2)), ("mcx3", lambda qc: qc.mcx([0, 1, 3], 4)),
                    ("mcx4", lambda qc: qc.mcx([4, 3, 1, 0], 2))]:
        qc = prep(n); f(qc); out.append(("S3", name, qc))
    for fam in ("ghz", "qft", "rqc", "vqe", "grover"):
        for n in range(2, max_n_fid + 1, 2):
            if fam == "grover" and n > 12:
                continue
            out.append(("S4", f"{fam}_{n}", make(fam, n)))
    return out


def large_checks(state, fam, n):
    """analytic check for large n (no reference statevector)."""
    if fam == "ghz":
        a0, a1 = state[0], state[-1]
        rest = np.sum(np.abs(state) ** 2) - abs(a0) ** 2 - abs(a1) ** 2
        return float(1 - (abs(a0) ** 2 + abs(a1) ** 2) + abs(abs(a0) ** 2 - 0.5) + abs(abs(a1) ** 2 - 0.5) + rest)
    if fam == "qft":
        return float(np.max(np.abs(np.abs(state) ** 2 - 1.0 / state.size)) * state.size)
    raise ValueError(fam)


def run_suite(run_fn, platform, out_path, thr=1e-5, max_n_fid=20, large_ns=(), supported=None):
    recs = []
    for sec, name, qc in cases(max_n_fid):
        t0 = time.perf_counter()
        try:
            sv = run_fn(qc)
            ref = Statevector(qc).data
            nrm = float(np.vdot(sv, sv).real)
            F = float(abs(np.vdot(ref, sv)) ** 2 / nrm)
            ok = (1 - F) <= thr and abs(nrm - 1) <= 1e-4
            err = None
        except Exception as e:  # unsupported gate etc.
            F, nrm, ok, err = None, None, False, f"{type(e).__name__}: {e}"[:200]
        recs.append(dict(section=sec, case=name, n=qc.num_qubits, fidelity=F, infidelity=None if F is None else 1 - F,
                         norm=nrm, passed=ok, error=err, wall_s=time.perf_counter() - t0))
        if not ok:
            print(f"FAIL {sec} {name} F={F} err={err}", flush=True)
    for fam in ("ghz", "qft"):
        for n in large_ns:
            qc = make(fam, n)
            t0 = time.perf_counter()
            sv = run_fn(qc)
            dev = large_checks(sv, fam, n)
            recs.append(dict(section="S5", case=f"{fam}_{n}_analytic", n=n, analytic_deviation=dev, passed=dev < 1e-4,
                             wall_s=time.perf_counter() - t0))
            print(f"S5 {fam}_{n} deviation {dev:.2e}", flush=True)
            del sv
    summ = dict(platform=platform, threshold=thr, total=len(recs), passed=sum(r["passed"] for r in recs),
                by_section={s: [sum(1 for r in recs if r["section"] == s and r["passed"]),
                                sum(1 for r in recs if r["section"] == s)] for s in ("S1", "S2", "S3", "S4", "S5")})
    json.dump(dict(summary=summ, records=recs), open(out_path, "w"), indent=1)
    print(json.dumps(summ))
    return summ
