# Code map

## 1. v07 baseline (submitted-era code, `opt_kernel/` originals; working copies in `host/v07/`, `kernels/v07/`)

| Stage | Where | What happens |
|---|---|---|
| Circuit intake | `fpga_simulator_v07.py: FPGASimulator.run` | Qiskit circuit -> gate tuples (name, qubits, params) |
| Split / measure / reset | `_validate_and_split_gates` | segments between resets; measure -> (qubit, clbit) pairs |
| **Lowering** | `_lower_to_kernel_basis` | every non-native gate -> {1q, CX, CZ}; e.g. CP -> P, P, CX, RZ, CX; SWAP -> 3 CX; ECR -> 11 gates; CCX -> 15; MCX via Qiskit transpile (`_decompose_mcx_qiskit`) |
| Encoding | `_encode_gates` | 8 x int32 per gate: opcode, q0, q1, q2, three float32 params (angles truncated to float32) |
| Chunking | `_execute_segment` | at most 512 gates per kernel launch (BRAM gate buffer) |
| Data layout | `_distribute_statevector` / `_collect_statevector` | complex128; amplitude i -> bank i & 7, slot i >> 3; 8 BOs, one per bank argument (3 PCs each, alveo_u55c.cfg) |
| Kernel | `quantum_fpga_kernel_unified.cpp` | 19 opcodes; per gate one pipelined pair loop (II=1 schedule, single-beat accesses via an 8-way runtime bank mux); `apply_cnot` iterates N/2 with two inserted bits (out-of-bounds, see BASELINE_AUDIT B) |
| Timing | `run_circuit` | `kernel_us` = gate upload + launch + wait (we split these, marked [jsc2026]); energy and "HBM traffic" are analytical (`EnergyEstimator`, `_profile_gates`), not measured |
| Limits | constants | MAX_QUBITS_FPGA = 28 (6 GB of bank BOs), KERNEL_MAX_GATES = 512 |

## 2. New SV stack (this revision)

| Stage | Where | What happens |
|---|---|---|
| Gate -> op | `host/sv/compiler.py: circuit_to_ops` | every 1q gate -> 2x2 matrix (fused per qubit); 2q diagonal/controlled -> one MUX2; other 2q -> 3 MUX2 (cosine-sine decomposition); multi-controlled 1q (CCX, CCZ, MCX) -> one MUX2; SWAP -> relabel (no data movement); only gates with no controlled-1q form (e.g. CSWAP) are lowered by Qiskit |
| Scheduling | `schedule` | greedy passes: ops whose qubits fit in the tile window (2^k amplitudes, up to 2 high local qubits) share one HBM sweep; bank qubits trigger remaps (fast path: bank <-> high local qubit) |
| Encoding | `encode_segment`, `isa.py` | QSV2 stream: header, pass headers (m, h0, h1, nops), 24-word MUX2 records |
| Golden model | `host/sv/model.py` | numpy execution of the exact stream, per PC slice and tile; validated vs Qiskit (`tests/test_compiler_model.py`) |
| Engine kernel | `kernels/sv/sv_engine.cpp` | one CU per PC; per pass DATAFLOW reader -> compute -> writer with parity-banked tile buffers, 4 amplitude pairs/cycle, II=1; instructions read from the same BO (single AXI port per CU) |
| Exchange kernel | `kernels/sv/sv_swap.cpp` | device-side block swap between PCs (ports through the HBM switch) for bank-qubit remaps |
| Runtime | `host/sv/runtime.py` | CU discovery, BO allocation, padding for small n, launches, remaps (device or host), Sec. 6 timing split, sampling |
| Builds | `builds/link_sv.sh`, `builds/sv_xo/` | one .xo, linked as 16 / 8 engine CUs (+4 swap CUs) |

## 3. Measurement code
`bench/fpga/{v07_bench,sv_bench,hbm_bench,pcie_probe}.py`, `bench/cpu/cpu_bench.py`, `bench/gpu/{gpu_bench.py,run_remote.sh}`,
`bench/power/fpga_power.py`, `tests/*` (unit, functional suite, csim), `analysis/*`.
