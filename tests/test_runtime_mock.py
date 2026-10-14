"""Host orchestration test on the mock pyxrt: SVRuntime (1 card, device and host swaps, layouts) and
MultiSVRuntime (2 and 4 cards) vs Qiskit."""
import os, sys
WS = os.environ.get("WS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))); sys.path[:0] = [f"{WS}/tests", f"{WS}/host/sv", f"{WS}/circuits"]
import mock_pyxrt; mock_pyxrt.install()
import numpy as np
from qiskit.quantum_info import Statevector
from runtime import SVRuntime
from multicard import MultiSVRuntime
from generators import make
from circuit_zoo import zoo
fails = 0; cases = 0
circs = [make(f, n) for f in ("ghz", "qft", "rqc", "vqe") for n in (3, 9, 12)] + [zoo(10, s) for s in range(3)]
for ne, ns, dev_swap in ((8, 4, True), (16, 4, True), (4, 0, False)):
    rt = SVRuntime("0000:41:00.1", "x", ne, ns, use_device_swap=dev_swap)
    for qc in circs:
        for k in (12, 5):
            out = rt.run(qc, k=k, shots=16)["state"]
            inf = 1 - abs(np.vdot(Statevector(qc).data, out)) ** 2
            cases += 1
            if inf > 1e-5:
                fails += 1; print("FAIL single", ne, ns, dev_swap, qc.name, k, inf)
for C in (2, 4):
    m = MultiSVRuntime(["0000:41:00.1", "0000:a1:00.1", "0000:41:00.1", "0000:a1:00.1"][:C], "x", 4, 2)
    for qc in [make(f, 12) for f in ("ghz", "qft", "rqc", "vqe")] + [zoo(11, 7)]:
        r = m.run(qc, k=6, shots=8)
        inf = 1 - abs(np.vdot(Statevector(qc).data, r["state"])) ** 2
        cases += 1
        if inf > 1e-5:
            fails += 1; print("FAIL multi", C, qc.name, inf)
        else:
            print("ok multi", C, qc.name, "remaps", r["stats"]["n_remaps"], "exch_dev %.3f host %.3f" % (r["timing"]["exchange_device_s"], r["timing"]["exchange_host_s"]))
# sampling check: 20000 shots of a 10-qubit RQC vs exact probabilities (total variation distance)
rt = SVRuntime("0000:41:00.1", "x", 8, 4)
qc = make("rqc", 10)
r = rt.run(qc, shots=20000, seed=3)
p = np.abs(Statevector(qc).data) ** 2
emp = np.zeros_like(p)
for i, c in r["counts"].items(): emp[i] = c / 20000
tvd = 0.5 * np.abs(emp - p).sum()
print("sampling TVD (20000 shots, 1024 outcomes):", round(tvd, 4))
cases += 1; fails += tvd > 0.12
print(f"cases={cases} fails={fails}")

# virtual slices: 4 engines x 2 PCs each (fake link.cfg) -> up to 8 single-PC slices
vb = f"{WS}/tmp/fakebuild/sv.xclbin"
rt = SVRuntime("0000:41:00.1", vb, 4, 2)
vfail = 0
for qc in [make(f, n) for f in ("ghz", "qft", "rqc", "vqe") for n in (13, 14, 15)] + [zoo(14, 3)]:
    r = rt.run(qc, shots=8)
    inf = 1 - abs(np.vdot(Statevector(qc).data, r["state"])) ** 2
    vfail += inf > 1e-5
print("virtual-slice runs: P max", rt.n_slots, "fails", vfail, "swap_ok", rt.swap_ok)
m = MultiSVRuntime(["0000:41:00.1", "0000:a1:00.1"], vb, 4, 2)
for qc in [make(f, 15) for f in ("qft", "rqc")]:
    r = m.run(qc, shots=8); inf = 1 - abs(np.vdot(Statevector(qc).data, r["state"])) ** 2
    vfail += inf > 1e-5; print("multi virtual", qc.name, "P", r["P"], "1-F %.1e" % inf)
print(f"virtual total fails={vfail}")
sys.exit(1 if fails or vfail else 0)
