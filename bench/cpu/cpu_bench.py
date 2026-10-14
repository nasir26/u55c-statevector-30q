"""CPU baselines on the benchmark circuits (same instances as the FPGA runs).

Backends
  numpy1t   : vectorised NumPy statevector, complex64, 1 thread (legacy reference;
              thread env vars must be set to 1 by the caller, recorded in output)
  aer       : Qiskit Aer statevector, --threads threads, precision single|double, fusion on|off
  qulacs    : Qulacs QuantumState (double; OpenMP threads = OMP_NUM_THREADS)
  qsim      : qsimcirq (single precision, --threads threads)

Times reported: T_kernel = compute only (statevector evolution); T_e2e adds circuit
construction/transpile (T_compile). Every record keeps the backend's own timer too.
"""
import argparse, json, math, os, platform, sys, time
import numpy as np

WS = os.environ.get("WS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.insert(0, f"{WS}/circuits")
from generators import make, to_tuples  # noqa: E402

# ---------------------------------------------------------------- numpy reference
_H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)


def _u1q(nm, p):
    c = np.complex64
    if nm == "h": return _H
    if nm == "x": return np.array([[0, 1], [1, 0]], c)
    if nm == "t": return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], c)
    if nm == "rx":
        a, b = math.cos(p[0] / 2), math.sin(p[0] / 2); return np.array([[a, -1j * b], [-1j * b, a]], c)
    if nm == "ry":
        a, b = math.cos(p[0] / 2), math.sin(p[0] / 2); return np.array([[a, -b], [b, a]], c)
    if nm == "rz":
        return np.array([[np.exp(-0.5j * p[0]), 0], [0, np.exp(0.5j * p[0])]], c)
    if nm == "p": return np.array([[1, 0], [0, np.exp(1j * p[0])]], c)
    raise KeyError(nm)


def np_apply_1q(psi, n, q, U):
    v = psi.reshape(1 << (n - q - 1), 2, 1 << q)
    a0 = v[:, 0, :].copy()
    a1 = v[:, 1, :]
    v[:, 0, :] = U[0, 0] * a0 + U[0, 1] * a1
    v[:, 1, :] = U[1, 0] * a0 + U[1, 1] * a1


def _view2(psi, n, a, b):
    """5-D view with axes for qubits hi=max(a,b) and lo=min(a,b)."""
    hi, lo = max(a, b), min(a, b)
    return psi.reshape(1 << (n - hi - 1), 2, 1 << (hi - lo - 1), 2, 1 << lo), hi, lo


def np_apply(psi, n, nm, qs, p):
    if nm in ("cx", "cz", "cp", "swap"):
        v, hi, lo = _view2(psi, n, qs[0], qs[1])
        def sl(bit_hi, bit_lo): return v[:, bit_hi, :, bit_lo, :]
        c, t = qs[0], qs[1]
        if nm == "cx":
            if c == hi: tmp = sl(1, 0).copy(); v[:, 1, :, 0, :] = sl(1, 1); v[:, 1, :, 1, :] = tmp
            else:       tmp = sl(0, 1).copy(); v[:, 0, :, 1, :] = sl(1, 1); v[:, 1, :, 1, :] = tmp
        elif nm == "cz": v[:, 1, :, 1, :] *= -1
        elif nm == "cp": v[:, 1, :, 1, :] *= np.complex64(np.exp(1j * p[0]))
        else: tmp = sl(0, 1).copy(); v[:, 0, :, 1, :] = sl(1, 0); v[:, 1, :, 0, :] = tmp
    else:
        np_apply_1q(psi, n, qs[0], _u1q(nm, p))


def run_numpy(gates, n):
    t0 = time.perf_counter()
    psi = np.zeros(1 << n, dtype=np.complex64); psi[0] = 1
    for nm, qs, p in gates:
        np_apply(psi, n, nm, qs, p)
    return psi, time.perf_counter() - t0


# ---------------------------------------------------------------- aer
def run_aer(qc, threads, precision, fusion):
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    sim = AerSimulator(method="statevector", precision=precision, max_parallel_threads=threads,
                       fusion_enable=fusion)
    t0 = time.perf_counter()
    q = qc.copy(); q.save_statevector()
    tq = transpile(q, sim, optimization_level=0)
    t1 = time.perf_counter()
    r = sim.run(tq).result()
    t2 = time.perf_counter()
    sv = np.asarray(r.get_statevector())
    return sv, dict(T_compile_s=t1 - t0, T_run_s=t2 - t1, aer_time_taken_s=r.results[0].time_taken,
                    aer_metadata=r.results[0].metadata)


# ---------------------------------------------------------------- qulacs
def run_qulacs(gates, n):
    from qulacs import QuantumState, QuantumCircuit as QC
    from qulacs.gate import DenseMatrix
    t0 = time.perf_counter()
    c = QC(n)
    for nm, qs, p in gates:
        if nm == "h": c.add_H_gate(qs[0])
        elif nm == "x": c.add_X_gate(qs[0])
        elif nm == "t": c.add_T_gate(qs[0])
        elif nm in ("rx", "ry", "rz"):  # qulacs rotation sign convention is opposite to Qiskit
            getattr(c, f"add_R{nm[1].upper()}_gate")(qs[0], -p[0])
        elif nm == "p": c.add_gate(DenseMatrix(qs[0], _u1q("p", p).astype(complex)))
        elif nm == "cx": c.add_CNOT_gate(qs[0], qs[1])
        elif nm == "cz": c.add_CZ_gate(qs[0], qs[1])
        elif nm == "swap": c.add_SWAP_gate(qs[0], qs[1])
        elif nm == "cp":
            g = DenseMatrix(qs[1], _u1q("p", p).astype(complex)); g.add_control_qubit(qs[0], 1); c.add_gate(g)
        else: raise KeyError(nm)
    st = QuantumState(n); st.set_zero_state()
    t1 = time.perf_counter()
    c.update_quantum_state(st)
    t2 = time.perf_counter()
    return st.get_vector(), dict(T_compile_s=t1 - t0, T_run_s=t2 - t1)


# ---------------------------------------------------------------- qsim
def run_qsim(gates, n, threads):
    import cirq, qsimcirq
    t0 = time.perf_counter()
    q = cirq.LineQubit.range(n)
    ops = []
    for nm, qs, p in gates:
        if nm == "h": ops.append(cirq.H(q[qs[0]]))
        elif nm == "x": ops.append(cirq.X(q[qs[0]]))
        elif nm == "t": ops.append(cirq.T(q[qs[0]]))
        elif nm == "rx": ops.append(cirq.rx(p[0])(q[qs[0]]))
        elif nm == "ry": ops.append(cirq.ry(p[0])(q[qs[0]]))
        elif nm == "rz": ops.append(cirq.rz(p[0])(q[qs[0]]))
        elif nm == "p": ops.append(cirq.ZPowGate(exponent=p[0] / math.pi)(q[qs[0]]))
        elif nm == "cx": ops.append(cirq.CNOT(q[qs[0]], q[qs[1]]))
        elif nm == "cz": ops.append(cirq.CZ(q[qs[0]], q[qs[1]]))
        elif nm == "swap": ops.append(cirq.SWAP(q[qs[0]], q[qs[1]]))
        elif nm == "cp": ops.append(cirq.CZPowGate(exponent=p[0] / math.pi)(q[qs[0]], q[qs[1]]))
        else: raise KeyError(nm)
    circ = cirq.Circuit(ops)
    sim = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=threads))
    t1 = time.perf_counter()
    r = sim.simulate(circ, qubit_order=q[::-1])  # big-endian order of reversed qubits = Qiskit order
    t2 = time.perf_counter()
    return np.asarray(r.final_state_vector), dict(T_compile_s=t1 - t0, T_run_s=t2 - t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["numpy1t", "aer", "qulacs", "qsim"])
    ap.add_argument("--families", default="ghz,qft,rqc,vqe")
    ap.add_argument("--ns", default="4,8,12,16,20,24")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--precision", default="double")
    ap.add_argument("--fusion", type=int, default=1)
    ap.add_argument("--check-max-n", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = []
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                          "OMP_PROC_BIND", "OMP_PLACES")}
    for fam in a.families.split(","):
        for n in [int(x) for x in a.ns.split(",")]:
            qc = make(fam, n)
            gates = to_tuples(qc)
            ref = None
            for r in range(a.reps):
                if a.backend == "numpy1t":
                    sv, t = run_numpy(gates, n); extra = dict(T_run_s=t)
                elif a.backend == "aer":
                    sv, extra = run_aer(qc, a.threads, a.precision, bool(a.fusion))
                elif a.backend == "qulacs":
                    sv, extra = run_qulacs(gates, n)
                else:
                    sv, extra = run_qsim(gates, n, a.threads)
                if r == 0 and n <= a.check_max_n:
                    from qiskit.quantum_info import Statevector
                    ref = Statevector(qc).data if n <= 20 else None
                fid = float(abs(np.vdot(ref, sv)) ** 2 / np.vdot(sv, sv).real) if ref is not None else None
                rec = dict(platform=f"cpu_{a.backend}", family=fam, n=n, rep=r, circuit=qc.name,
                           threads=a.threads if a.backend != "numpy1t" else 1, precision=a.precision,
                           fusion=a.fusion, fidelity_vs_qiskit=fid, **{k: v for k, v in extra.items() if k != "aer_metadata"})
                if "aer_metadata" in extra:
                    md = extra["aer_metadata"]
                    rec["aer_parallel_state_update"] = md.get("parallel_state_update")
                    rec["aer_fusion"] = md.get("fusion")
                recs.append(rec)
                print(f"{a.backend:7s} {qc.name:14s} rep{r} run {rec['T_run_s']:.4f}s F={fid}", flush=True)
                del sv
            with open(a.out, "w") as f:
                json.dump(dict(host=platform.node(), env=env, args=vars(a), records=recs), f, indent=1, default=str)


if __name__ == "__main__":
    main()
