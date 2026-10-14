# Baseline audit (v07 kernel), working notes

## A. Harness pitfall: `import uuid` after XRT device init (XRT 2.16, pyxrt, conda py3.8)
- Symptom: after a pyxrt device is opened and the first launch is done, importing Python's `uuid`
  module (directly, or via qiskit / qiskit_aer) makes **every later launch report
  ERT_CMD_STATE_COMPLETED while leaving HBM unchanged**. Importing it before device init is fine;
  `time.sleep`, rustworkx, symengine, scipy and multiprocessing imports are fine.
- Evidence: `tests/uuid_probe.py` (standalone, X gate on |0>), output in `logs/phase1/uuid_probe.txt`: modes none/uuid_first all ok; uuid/qiskit imported after first launch, then every later launch is NOOP.
- Impact on the submitted results: none found. `fpga_simulator_v07.py` imports uuid and qiskit at
  module top, before any device is created.
- Mitigation: all our host code imports uuid/qiskit before pyxrt init, and the new host adds a
  per-launch canary (a kernel-side cycle counter / status word) so a silent no-op launch cannot go unnoticed.

## B. Native opcode unit tests (v07 xclbin, card 41:00.1, n = 6)
Raw: `results/raw/phase1/v07_opcode_unit_n6.json` (harness `tests/v07_direct.py`, seed 1234).
324 cases: 17 single-qubit opcodes x 6 positions and CX/CZ x 30 ordered pairs, each on a random basis state and a Haar state.
Reference: Qiskit `Statevector.evolve`, little-endian, angles cast to float32 as the kernel receives them.
- All single-qubit opcodes and CZ: max |error| 2.5e-16, worst infidelity 2.2e-16 (complex128 round-off).
- **CX: final state correct in 60/60 cases, but every CX writes outside the statevector** (64 of the
  guard words changed at n = 6, i.e. N/2 amplitude words beyond the state across the 8 banks).
  Cause (kernels/v07/quantum_fpga_kernel_unified.cpp, `apply_cnot`): the loop runs N/2 iterations but
  inserts two zero bits (min_bit, max_bit), so `p_high` carries one extra bit and indices reach 2N.
  The upper half of the iterations swaps amplitude pairs that lie wholly beyond the state.
  Consequences: (i) CX costs 2x the necessary HBM traffic; (ii) out-of-bounds HBM reads/writes past
  each bank BO (the v07 host sizes BOs exactly); (iii) at n = 30, indices up to 2^31 overflow `int`.

## C. v07 timing baseline (card 41:00.1; raw `results/raw/phase1/v07_baseline_*.json`, summary `results/processed/v07_baseline.csv`, script `analysis/v07_baseline_summary.py`)
- Grid: GHZ/QFT/RQC(d=10)/VQE(L=2), n = 4..24, 3 reps (median), plus GHZ 26 and 28 (1 rep). All fidelities vs Aer double >= 1 - 2e-13.
- **The kernel dominates**: T_kernel is >= 72.7% of T_e2e at n = 16 and 85-99.8% for n >= 20 (correction 2026-10-06: an
  earlier version of this note said 90-99.8% for n >= 16). H2D/D2H/compile/post are minor (GHZ-28: kernel 122.5 s, D2H 6.6 s, H2D 2.4 s, e2e 136.1 s).
- **Effective HBM throughput is about 2.0 GB/s, flat from n = 12 to 28** (bytes moved / T_kernel), i.e. about 0.4% of the 460 GB/s
  peak, about 10 kernel cycles per amplitude pair at 300 MHz instead of the 1 cycle the II=1 schedule promises.
  The time is consistent with un-bursted single-beat 64-bit AXI accesses (re and im read and written separately through a runtime
  8-way bank mux). To be confirmed against the hbm_stream ceilings (Phase 2: copy_nonburst vs copy_burst).
- Per kernel gate at n = 28: 4.37 s. v07 cannot run n > 28 (6 GB of bank BOs, complex128).
- QFT is the worst family because every CP is lowered into 5 kernel gates (p, p, cx, rz, cx), and each SWAP into 3 CX.
- The submitted GHZ-30 figure (22,686 s) was produced by a different kernel that is not available; it cannot be re-run.
  With the v07 per-gate scaling (4x per 2 qubits), GHZ-30 would need roughly 0.5 ks if it fitted [ESTIMATE by extrapolation, NOT measured; must not appear in the paper as a result], i.e. even v07 is far below 22.7 ks.

## D. v07 host lowerings are exact; the original functional suite is weak
- `tests/v07_lowering_check.py` (raw `results/raw/phase1/v07_lowering_check.json`): every lowering in
  `_lower_to_kernel_basis` (CY, CH, SWAP, CP, CRX/Y/Z, CSX, DCX, iSWAP, ECR, RXX, RYY, RZZ incl. theta = pi/2, CCX),
  both qubit orders, has process fidelity 1.000000 vs the Qiskit unitary. So in v07 ECR and RZZ are correct (by host
  lowering, not natively). The submitted ECR F = 0 / RZZ cos^2(pi/8) defects belong to the earlier kernel with native
  apply_ecr/RZZ, which is not available for re-test.
- The v07 functional suite (`comprehensive_test_v07.py`) has **91 tests, not 221**; on card 41 all 91 pass in 41.1 s
  (raw `results/raw/phase1/v07_suite/`). Its pass criterion is a norm check plus coarse probability checks
  (tolerance 0.02), not a fidelity vs a reference. 13 tests check the norm only (CH, CRZ, CSX, DCX, ECR, RXX, RYY,
  RZZ...), so a wrong unitary would pass. Grover n > 6 passes on norm only, with iterations capped below optimal.
- Grover wall time in v07 peaks at 3.9 s (n = 12), not ~574 s; the 574 s figure is not reproducible with any
  available code. The revision replaces the suite with a fidelity-based one (tests/functional_suite.py).

## E. CUDA-Q state ordering (relevant to R2.5, Fig. 2)
CUDA-Q 0.12 `get_state` returns the statevector in little-endian order (X on qubit 0 of 3 -> index 1;
auto-detected and logged by bench/gpu/gpu_bench.py). A big-endian assumption leaves GHZ and QFT|0>
(both symmetric under qubit reversal) apparently correct but gives F ~ 0.06 (n=4) down to 4.5e-6 (n=20)
for RQC (observed in our first smoke test, results/raw/gpu/smoke_cudaq run 1 before the fix). This is a
plausible cause of the inconsistent GPU fidelity panel in the submitted Fig. 2; the original GPU script is
not available, so this is a hypothesis, not a confirmed diagnosis.

## F. New-kernel hardware bring-up finding (sv_engine v1)
First hardware run (sv_e8 v1 on card a1) was correct for tile size k >= 9 and wrong for k <= 8 (sweep in session log).
Cause: HLS flattened the OPS and PAIRS loops into one pipeline (depth 32); with the DEPENDENCE pragmas the next op read
words the previous op had not yet written. C-simulation is sequential and cannot show this. Fix: LOOP_FLATTEN off on
OPS (pipeline drains between ops, ~29 cycles per op). The v1 bitstream is kept as builds/sv_e8_v1_flatbug for the record;
all results use the fixed kernel.

## G. Fidelity suite on the new kernel (sv_e8, fixed build, card a1)
396/396 pass at 1-F <= 1e-5 (raw results/raw/phase4/sv_functional_sv_e8.json): S1 85/85, S2 252/252 (all 21 two-qubit
gates incl. ECR, iSWAP, RXX/RYY/RZZ, XX+YY, both orders, every pair), S3 7/7 (CCX, CCZ, CSWAP, MCX 3/4 controls),
S4 46/46 (families incl. Grover to n=12), S5 6/6 analytic GHZ/QFT at n = 24, 26, 28. v07: 328/396 (unsupported gates).

## H. XRT memory-group trap (multi-PC engine buffers + device swap kernel)
On sv_e16w (engine ports on HBM[2i:2i+1]) buffers allocated in the engine's memory group live in a 2-PC group bank
that the sv_swap ports (HBM[0:31]) do not share. XRT then *silently allocates a private copy* for the swap kernel
("Allocating local copy of argument buffer in connected bank"), the swap acts on the copy, and the state becomes wrong
(suite: 378/398; GHZ-18..30 garbage). Host-mediated swaps were exact. Fix in host/sv/runtime.py: engine buffers are
allocated in an explicit PC bank (from the build's link.cfg) whenever the slice fits in one 512 MB PC, and a device-swap
self-test on the real buffers runs at every allocation; when buffers must span two PCs (n = 30 on 16 CUs) the runtime
uses host-mediated exchange and logs it.
