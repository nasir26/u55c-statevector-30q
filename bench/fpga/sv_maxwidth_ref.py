"""Reference check for bench/fpga/sv_maxwidth.py: complex128 Qiskit Aer statevector (fusion off, all threads) for
each saved FPGA state; fidelity, infidelity without cancellation (1 - F = d^2 - d^4/4, d the l2 distance after
optimal global phase alignment of the normalised states) and the largest amplitude error. Run with the card idle.
"""
import argparse, json, os, sys, time
import numpy as np

WS = os.environ.get("WS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path[:0] = [f"{WS}/circuits"]
from generators import make  # noqa: E402

CH = 1 << 24


def metrics(ref, sv):
    nr = np.sqrt(sum(float(np.vdot(ref[i:i + CH], ref[i:i + CH]).real) for i in range(0, len(ref), CH)))
    ns = np.sqrt(sum(float(np.vdot(sv[i:i + CH].astype(np.complex128), sv[i:i + CH].astype(np.complex128)).real)
                     for i in range(0, len(sv), CH)))
    ip = sum(complex(np.vdot(ref[i:i + CH], sv[i:i + CH].astype(np.complex128))) for i in range(0, len(ref), CH)) / (nr * ns)
    ph = ip / abs(ip)
    d2, mx = 0.0, 0.0
    for i in range(0, len(ref), CH):
        e = sv[i:i + CH].astype(np.complex128) / ns - ph * ref[i:i + CH] / nr
        d2 += float(np.vdot(e, e).real); mx = max(mx, float(np.abs(e).max()))
    return dict(fidelity=abs(ip) ** 2, one_minus_f=d2 - d2 * d2 / 4, l2=float(np.sqrt(d2)), max_abs_err=mx, norm_fpga=ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--state-dir", required=True)
    a = ap.parse_args()
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    import qiskit_aer
    sim = AerSimulator(method="statevector", precision="double", fusion_enable=False)
    d = json.load(open(a.run))
    for rec in d["records"]:
        if not rec.get("state_file"):
            continue
        qc = make(rec["family"], rec["n"], depth=rec["depth"]) if rec["family"] == "rqc" else make(rec["family"], rec["n"])
        q = qc.remove_final_measurements(inplace=False); q.save_statevector()
        t0 = time.perf_counter()
        ref = np.asarray(sim.run(transpile(q, sim, optimization_level=0)).result().get_statevector().data)
        tr = time.perf_counter() - t0
        sv = np.load(os.path.join(a.state_dir, rec["state_file"]), mmap_mode="r")
        m = metrics(ref, sv)
        rec.update(m, ref_kind=f"qiskit_aer {qiskit_aer.__version__} statevector double, fusion off", ref_s=tr)
        print(f"{rec['family']:4s} n={rec['n']:2d}  1-F={m['one_minus_f']:.2e}  max|da|={m['max_abs_err']:.2e}  "
              f"|psi|={m['norm_fpga']:.7f}  ref {tr:.1f}s", flush=True)
        del ref, sv
        json.dump(d, open(a.run, "w"), indent=1)


if __name__ == "__main__":
    main()
