"""Run the fidelity suite on an SV kernel build. usage: run_suite_sv.py <bdf> <xclbin> <engines> <swaps> <out.json> [large_ns]"""
import uuid  # noqa: F401
import os, sys
WS = os.environ.get("WS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))); sys.path[:0] = [f"{WS}/host/sv", f"{WS}/tests", f"{WS}/circuits"]
from runtime import SVRuntime
from functional_suite import run_suite
bdf, xcl, ne, ns, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
large = tuple(int(x) for x in sys.argv[6].split(",")) if len(sys.argv) > 6 else ()
rt = SVRuntime(bdf, xcl, ne, ns)
def run(qc):
    return rt.run(qc, shots=0, return_state=True)["state"].astype(complex)
run_suite(run, "fpga_" + os.path.basename(os.path.dirname(os.path.abspath(xcl))), out, thr=1e-5, max_n_fid=20, large_ns=large)
