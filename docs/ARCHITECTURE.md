# SV kernel architecture (draft v1, 2026-10-05; to be revised with Phase 2 measurements)

## Why v07 is slow (from Phase 1)
v07 issues one amplitude pair per loop iteration with single-beat 64-bit accesses (re and im separately)
through a runtime 8-way bank multiplexer. The measured 2.0 GB/s (about 10 cycles per pair) is a
memory-level-parallelism problem: no bursts and few outstanding requests. HBM bandwidth and bank
conflicts are not the limit. The new design therefore streams long bursts per pseudo-channel and moves
all irregular indexing on chip.

## Data format and layout
- complex64 (float32 re, im interleaved); one 512-bit word = 8 amplitudes.
- **Block layout**: with P_act active PCs (runtime, power of two, at most P), the top b = log2(P_act) qubits of the
  physical index select the PC. Each PC holds a contiguous slice of 2^(n-b) amplitudes. Qubits below n-b are
  *local*; the top b are *bank qubits*.
  n = 30, P_act = 32: 256 MB per PC (PC capacity 512 MB). n = 31 needs 512 MB per PC, to be tested.
- Logical-to-physical qubit mapping is dynamic (host-side). A gate on a bank qubit triggers a remap that swaps
  bank and local qubits (a data permutation between PCs, see "Exchange"). "Modulo-P interleave" layouts (v07 style)
  are the same as mapping the *low* logical qubits onto bank positions, so the layout ablation is a choice of
  initial mapping plus remap policy, measured on the same hardware.

## Engine (one per PC; P in {8, 16, 32} structural builds)
Each engine processes one "pass" over its PC slice as a DATAFLOW pipeline:

    reader (bursts) -> tile buffer (ping-pong, URAM/BRAM) -> compute (op list) -> writer (bursts)

- Tile = 2^k amplitudes (k = 13: 1024 words, 64 KB). A pass has up to m = 2 *high* local qubits h0, h1 >= k-m.
  The tile is gathered as 2^m sub-blocks of 2^(k-m) contiguous amplitudes at offsets a*2^h0 + b*2^h1, so each
  sub-block is a long burst. Tile qubit j < k-m is physical qubit j; tile qubits k-m.. are h0, h1.
- Op list (per pass, applied to every tile in order): `U2(t, M)`: 2x2 complex on tile qubit t;
  `CU2(c, t, M)`: controlled 2x2 (CX, CY, CH, CRx/y/z, CSX...); `D2(a, b, d00..d11)`: two-qubit diagonal (CZ,
  CP, RZZ, CRZ); `U4(a, b, M4)`: general 4x4 (iSWAP, ECR, RXX, RYY, SWAP, fused 2q blocks).
  Throughput target: U2/CU2/D2 one word pair (8 amplitude pairs) per 2 cycles; U4 at half that rate.
- Pass I/O at a PC's ~14 GB/s takes about 2.7k cycles per 64 KB tile at 300 MHz, so several ops per pass are free
  before the engine becomes compute-bound. This is where gate fusion and tiling pay off.

## Host compiler (host/sv/compiler.py)
1. Qiskit circuit -> list of (op, logical qubits, matrix). Every 1q gate becomes a U2 matrix (no decomposition);
   2q gates map to CU2/D2/U4 by structure; 3q+ gates (CCX, MCX) are the only lowered ones (reported).
2. Optional host fusion: consecutive 1q gates on the same qubit are multiplied into one U2.
3. Pass scheduling: greedy. Keep appending ops while their qubits fit in {tile qubits} + up to 2 high local
   qubits; otherwise close the pass. A bank qubit triggers a remap first.
4. Encoding: fixed-size instruction records (pass headers + op records, matrices as float32).

A numpy executor of the same instruction stream (`host/sv/model.py`) is the golden model used to validate the
compiler (vs Qiskit) and the HLS C-simulation (vs the model) before any hardware build.

## Exchange (bank-qubit remap)
Device-side when available: a pass type in which engine p reads the half-slices it needs from partner PC p^2^j
(requires ports connected through the HBM switch; ablation vs local-only). Fallback: host-mediated permutation
(D2H, numpy transpose, H2D). The same mechanism with "card" in place of "PC" gives the multi-card design.

## Structural variants planned
B1: P = 32 engines; B2: P = 16; B3: P = 8; B4: P = 4. Runtime parameters: P_act, k_eff (tile size <= 2^k),
fusion on/off and pass scheduling on/off (host-side), initial mapping / layout.

## Revisions after the first build attempts (2026-10-06)
1. **Single primitive.** The engine executes only MUX2(t, cmask, M0, M1). Every 1q gate, every controlled-1q gate
   (any number of controls in the tile window), every 2q diagonal gate maps to one MUX2; any other 2q unitary is
   three MUX2 via the cosine-sine decomposition (exact). SWAP is a relabel. Only gates without a controlled-1q form
   and with >2 qubits (e.g. CSWAP) are decomposed by Qiskit on the host.
2. **One CU per engine, one AXI port per CU.** A single kernel with 32 HBM ports failed placement (121.5% BRAM in
   SLR0, 24,131 SLLs needed vs 23,040 available; hbm_stream first build). The SV design is therefore a small
   single-port kernel instantiated N times (`nk=sv_engine:N`), with CUs spread over SLR0-2 and 256-bit AXI.
   Instructions live in the same buffer as the state slice (the HBM subsystem has 33 port connections; a second
   port per CU exceeded it: "out of port connections on /hmss_0").
3. **Device-side exchange.** `xrt::bo::copy` between HBM banks is not supported by this shell (falls back to host
   copies at 3.7 GB/s, results/raw/phase2/bo_copy_probe_41.jsonl) and PCIe is ~10-11 GB/s per direction. Bank-qubit
   remaps therefore use `sv_swap` CUs whose two ports reach all 32 PCs through the HBM switch; the scheduler prefers
   swapping a bank qubit with a high local qubit so the exchanged chunks are >= 2^(L-6) amplitudes.
4. **Capacity.** With one PC (512 MB) per CU, 16 CUs hold 8 GB: a 30-qubit complex64 state would fill the PCs with no
   room for the instruction region. Variant `sv_e16w` maps each CU to two PCs (HBM[2i:2i+1], contiguous), giving 32 PCs
   / 16 GB with identical engines; e16 vs e16w isolates the 16-vs-32 PC question (R3.10).

| Build | Engine CUs | PCs used | Swap CUs | Max n (1 card) | Notes |
|---|---|---|---|---|---|
| sv_e8 | 8 | 8 (HBM[0,4,...,28]) | 4 | 28 | engine-count ablation |
| sv_e16 | 16 | 16 (even PCs) | 4 | 29 | 16-PC point |
| sv_e16w | 16 | 32 (pairs) | 4 | 30 | flagship; 31 on 2 cards, 32 on 4 cards |
| sv_e16w512 | 16 (512-bit ports) | 32 (pairs) | 4 | 30 | port-width ablation (m<=2) |
| sv_e16w_m4 | 16 | 32 (pairs) | 4 | 30 | **flagship**: up to 4 high qubits per pass (MAXH=4), ~2x fewer passes |

Placed utilisation (sv_e16w512): one engine CU = 26.5k LUT / 37.6k FF / 53 BRAM / 296 DSP, one swap CU = 7.4k LUT /
45 BRAM / 16 URAM; the HLS estimate (62k LUT) is ~2.3x pessimistic. The 512-bit port brings no bandwidth gain while a CU
streams one PC at a time (hbm1: ~12.7 GB/s per PC for 256- and 512-bit ports alike), so the flagship keeps 256-bit ports.
Measured ceilings: 32 PCs copy 399 GB/s (12.5 GB/s per PC); a design streaming 16 PCs at a time is bounded at ~200 GB/s.
