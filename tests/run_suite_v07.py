"""Run the fidelity suite on the v07 baseline (complex128 kernel, host lowering)."""
import uuid  # noqa: F401
import os, sys
WS = os.environ["WS"]; sys.path[:0] = [f"{WS}/host/v07", f"{WS}/tests", f"{WS}/circuits"]
import fpga_simulator_v07 as v07
from functional_suite import run_suite
from generators import to_tuples
eng = v07.QuantumFPGACircuit(f"{WS}/opt_kernel/opt_kernel/quantum_fpga_kernel_complex128.xclbin", device_index=sys.argv[1])
def run(qc):
    return eng.run_circuit(qc.num_qubits, to_tuples(qc), shots=0, validate=False)["statevector"]
run_suite(run, "fpga_v07", sys.argv[2], thr=1e-10, max_n_fid=20, large_ns=(24, 26, 28))
