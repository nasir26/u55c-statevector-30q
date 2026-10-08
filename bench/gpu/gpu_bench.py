"""GPU baselines on the benchmark circuits (same generators/seeds as FPGA and CPU runs).

Backends
  aer_gpu : Qiskit Aer statevector on GPU (precision single|double, cuStateVec on|off)
  cudaq   : CUDA-Q 'nvidia' target (fp32) or 'nvidia' with option fp64
Timings per circuit (revision Sec. 6):
  T_compile : Qiskit transpile (Aer) / CUDA-Q kernel construction
  T_run     : backend execution incl. state kept on device (Aer: result.time_taken also recorded;
              CUDA-Q: cudaq.get_state, synchronous)
  T_d2h     : state copy to host numpy (CUDA-Q) / included in T_run for Aer (it returns host memory)
Fidelity vs Qiskit Statevector (complex128, CPU) for n <= --fid-max-n.
Power: nvidia-smi power.draw sampled every 100 ms in a background thread; each record carries the mean
power over its run window, and with --energy-loop S each circuit is also repeated for >= S seconds.
Results are printed as lines starting with 'RESULT ' (JSON), so they can be collected from srun stdout.
"""
import argparse, json, math, os, subprocess, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generators import make, to_tuples  # noqa: E402  (shipped alongside)


class GpuPower:
    def __init__(self, idx=0, period_ms=100):
        self.samples = []
        self.p = subprocess.Popen(["nvidia-smi", "-i", str(idx), "--query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active",
                                   "--format=csv,noheader,nounits", f"-lms={period_ms}"], stdout=subprocess.PIPE, text=True)
        self.th = threading.Thread(target=self._rd, daemon=True); self.th.start()

    def _rd(self):
        for line in self.p.stdout:
            try:
                parts = [x.strip() for x in line.split(",")]
                self.samples.append((time.perf_counter(), float(parts[1]), float(parts[2]), float(parts[3]), parts[4]))
            except Exception:
                pass

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return dict(mean_w=None, n=0)
        return dict(mean_w=float(np.mean([x[1] for x in s])), n=len(s), sm_mhz=float(np.median([x[2] for x in s])),
                    max_temp_c=float(max(x[3] for x in s)), throttle=sorted(set(x[4] for x in s)))

    def stop(self):
        self.p.terminate()


def run_aer(qc, precision, custatevec):
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    sim = AerSimulator(method="statevector", device="GPU", precision=precision, cuStateVec_enable=custatevec)
    t0 = time.perf_counter()
    q = qc.copy(); q.save_statevector()
    tq = transpile(q, sim, optimization_level=0)
    t1 = time.perf_counter()
    r = sim.run(tq).result()
    sv = np.asarray(r.get_statevector())
    t2 = time.perf_counter()
    if not r.success:
        raise RuntimeError(str(r.status))
    md = r.results[0].metadata
    keep = {k: md.get(k) for k in ("method", "device", "precision", "cuStateVec_enable", "batched_shots_optimization",
                                   "parallel_state_update", "fusion", "num_qubits") if k in md}
    return sv, dict(T_compile_s=t1 - t0, T_run_s=t2 - t1, T_d2h_s=0.0, aer_time_taken_s=r.results[0].time_taken,
                    aer_meta=keep, sv_dtype=str(sv.dtype))


def cudaq_kernel(gates, n):
    import cudaq
    k = cudaq.make_kernel()
    q = k.qalloc(n)
    for nm, qs, p in gates:
        if nm == "h": k.h(q[qs[0]])
        elif nm == "x": k.x(q[qs[0]])
        elif nm == "t": k.t(q[qs[0]])
        elif nm == "rx": k.rx(p[0], q[qs[0]])
        elif nm == "ry": k.ry(p[0], q[qs[0]])
        elif nm == "rz": k.rz(p[0], q[qs[0]])
        elif nm == "p": k.r1(p[0], q[qs[0]])
        elif nm == "cx": k.cx(q[qs[0]], q[qs[1]])
        elif nm == "cz": k.cz(q[qs[0]], q[qs[1]])
        elif nm == "cp": k.cr1(p[0], q[qs[0]], q[qs[1]])
        elif nm == "swap": k.swap(q[qs[0]], q[qs[1]])
        else: raise KeyError(nm)
    return k


def run_cudaq(qc, fp64):
    import cudaq
    t0 = time.perf_counter()
    k = cudaq_kernel(to_tuples(qc), qc.num_qubits)
    t1 = time.perf_counter()
    st = cudaq.get_state(k)
    t2 = time.perf_counter()
    sv = np.array(st, copy=True)
    t3 = time.perf_counter()
    n = qc.num_qubits
    if CUDAQ_ORDER == "big":   # convert to Qiskit little-endian (qubit 0 = least significant bit)
        sv = sv.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)
    return sv, dict(T_compile_s=t1 - t0, T_run_s=t2 - t1, T_d2h_s=t3 - t2)


CUDAQ_ORDER = None


def detect_cudaq_order():
    """X on qubit 0 of 3: index 1 -> little-endian, index 4 -> big-endian."""
    import cudaq
    k = cudaq.make_kernel(); q = k.qalloc(3); k.x(q[0])
    i = int(np.argmax(np.abs(np.array(cudaq.get_state(k)))))
    return {1: "little", 4: "big"}[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["aer_gpu", "cudaq"])
    ap.add_argument("--families", default="ghz,qft,rqc,vqe")
    ap.add_argument("--ns", default="4,8,12,16,20,24")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--precision", default="single")
    ap.add_argument("--custatevec", type=int, default=1)
    ap.add_argument("--fid-max-n", type=int, default=24)
    ap.add_argument("--energy-loop", type=float, default=0.0)
    a = ap.parse_args()
    if a.backend == "cudaq":
        import cudaq
        cudaq.set_target("nvidia", option="fp64" if a.precision == "double" else "fp32")
        global CUDAQ_ORDER
        CUDAQ_ORDER = detect_cudaq_order()
        print("RESULT " + json.dumps(dict(meta="cudaq_state_order", order=CUDAQ_ORDER)), flush=True)
    pw = GpuPower()
    time.sleep(1.0)
    from qiskit.quantum_info import Statevector
    for fam in a.families.split(","):
        for n in [int(x) for x in a.ns.split(",")]:
            qc = make(fam, n)
            ref = None
            for r in range(a.reps + 1):          # rep 0 = warm-up (recorded, flagged)
                t0 = time.perf_counter()
                try:
                    if a.backend == "aer_gpu":
                        sv, T = run_aer(qc, a.precision, bool(a.custatevec))
                    else:
                        sv, T = run_cudaq(qc, a.precision == "double")
                    err = None
                except Exception as e:
                    sv, T, err = None, {}, f"{type(e).__name__}: {e}"[:300]
                t1 = time.perf_counter()
                fid = None
                if sv is not None and n <= a.fid_max_n:
                    if ref is None:
                        ref = Statevector(qc).data
                    fid = float(abs(np.vdot(ref, sv)) ** 2 / np.vdot(sv, sv).real)
                rec = dict(platform=f"gpu_{a.backend}", family=fam, n=n, rep=r - 1, warmup=(r == 0), precision=a.precision,
                           custatevec=a.custatevec, fidelity=fid, error=err, wall_s=t1 - t0, **T, power=pw.window(t0, t1))
                print("RESULT " + json.dumps(rec), flush=True)
                del sv
                if err:
                    break
            if a.energy_loop > 0 and err is None:
                t0 = time.perf_counter(); cnt = 0
                while time.perf_counter() - t0 < a.energy_loop:
                    if a.backend == "aer_gpu":
                        run_aer(qc, a.precision, bool(a.custatevec))
                    else:
                        run_cudaq(qc, a.precision == "double")
                    cnt += 1
                t1 = time.perf_counter()
                print("RESULT " + json.dumps(dict(platform=f"gpu_{a.backend}", family=fam, n=n, energy_loop=True,
                                                   precision=a.precision, custatevec=a.custatevec, runs=cnt,
                                                   loop_s=t1 - t0, per_run_s=(t1 - t0) / cnt, power=pw.window(t0, t1))), flush=True)
    pw.stop()


if __name__ == "__main__":
    main()
