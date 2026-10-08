"""Time the v07 baseline (original host, instrumented copy) on the benchmark families.

Timing fields (Sec. 6 of the revision prompt):
  T_compile = host lowering/decompose (decompose_us)
  T_h2d     = statevector init upload (t_init_us) + gate-buffer uploads (t_upload_us)
  T_kernel  = sum of XRT launch->wait over all 512-gate chunks (t_launch_us)
  T_d2h     = statevector read-back + de-interleave (collect_us)
  T_post    = sampling 1024 shots (measure_us)
Fidelity vs Aer statevector (complex128) is computed outside the timed region.
"""
import uuid  # noqa: F401  must precede pyxrt device init (docs/BASELINE_AUDIT.md A)
import argparse, json, os, sys, time, platform
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile

WS = os.environ["WS"]
sys.path.insert(0, f"{WS}/host/v07"); sys.path.insert(0, f"{WS}/circuits")
import fpga_simulator_v07 as v07  # noqa: E402
from generators import make, to_tuples, export_qasm  # noqa: E402


def aer_sv(qc):
    q = qc.copy(); q.save_statevector()
    sim = AerSimulator(method="statevector", precision="double")
    return np.asarray(sim.run(transpile(q, sim)).result().get_statevector())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdf", required=True)
    ap.add_argument("--families", default="ghz,qft,rqc,vqe")
    ap.add_argument("--ns", default="4,8,12,16,20,24")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fid-max-n", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.bdf[5:7] in ("01", "21"):
        sys.exit("cards 01/21 not allowed")
    xcl = f"{WS}/opt_kernel/opt_kernel/quantum_fpga_kernel_complex128.xclbin"
    t0 = time.perf_counter()
    eng = v07.QuantumFPGACircuit(xcl, device_index=a.bdf)
    t_setup = time.perf_counter() - t0
    recs = []
    os.makedirs(f"{WS}/circuits/qasm", exist_ok=True)
    for fam in a.families.split(","):
        for n in [int(x) for x in a.ns.split(",")]:
            qc = make(fam, n)
            export_qasm(qc, f"{WS}/circuits/qasm/{qc.name}.qasm")
            gates = to_tuples(qc)
            ref = aer_sv(qc) if n <= a.fid_max_n else None
            for r in range(a.reps):
                res = eng.run_circuit(n, gates, shots=1024, validate=False, circuit_name=qc.name)
                t = res["timing"]
                sv = res["statevector"]
                fid = float(abs(np.vdot(ref, sv)) ** 2) if ref is not None else None
                rec = dict(platform="fpga_v07", bdf=a.bdf, family=fam, n=n, rep=r, circuit=qc.name,
                           n_gates_in=len(gates), n_gates_kernel=res.get("num_gates", None),
                           T_compile_us=t["decompose_us"], T_init_us=eng._t_init_us,
                           T_gate_upload_us=eng._t_upload_us, T_kernel_us=eng._t_launch_us,
                           T_d2h_us=t["collect_us"], T_post_us=t["measure_us"],
                           wall_us=res["execution_time"] * 1e6, fidelity=fid)
                rec["T_h2d_us"] = rec["T_init_us"] + rec["T_gate_upload_us"]
                rec["T_e2e_us"] = rec["T_compile_us"] + rec["T_h2d_us"] + rec["T_kernel_us"] + rec["T_d2h_us"] + rec["T_post_us"]
                recs.append(rec)
                print(f"{qc.name:14s} rep{r} kernel {rec['T_kernel_us']/1e6:9.4f}s e2e {rec['T_e2e_us']/1e6:9.4f}s "
                      f"h2d {rec['T_h2d_us']/1e6:7.4f}s d2h {rec['T_d2h_us']/1e6:7.4f}s F={fid}", flush=True)
                with open(a.out, "w") as f:
                    json.dump(dict(t_setup_s=t_setup, host=platform.node(), records=recs), f, indent=1)


if __name__ == "__main__":
    main()
