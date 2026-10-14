"""Summarise the v07 baseline runs: medians, time breakdown, per-kernel-gate time, effective HBM BW.

Kernel gate counts come from the v07 host's own lowering pass (_lower_to_kernel_basis).
HBM bytes per kernel gate (complex128, 16 B/amp, read + write of every touched amplitude):
  1q gate: 2 * 2^n * 16 B; CX (v07 loop, incl. the out-of-bounds half): 2 * 2^n * 16 B,
  of which only half is the useful state; CZ: 2 * 2^(n-2) * 16 B.
Output: results/processed/v07_baseline.csv and a printed table.
"""
import json, os, sys
from collections import Counter
import numpy as np

WS = os.environ.get("WS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.insert(0, f"{WS}/host/v07"); sys.path.insert(0, f"{WS}/circuits")
import fpga_simulator_v07 as v07  # noqa: E402
from generators import make, to_tuples  # noqa: E402

recs = []
for f in ("v07_baseline_grid.json", "v07_baseline_ghz_large.json"):
    recs += json.load(open(f"{WS}/results/raw/phase1/{f}"))["records"]

rows = []
for key in sorted({(r["family"], r["n"]) for r in recs}, key=lambda k: (k[0], k[1])):
    rs = [r for r in recs if (r["family"], r["n"]) == key]
    fam, n = key
    segs, _, _ = v07._validate_and_split_gates(to_tuples(make(fam, n)))
    kg = [g for s in segs for g in v07._lower_to_kernel_basis(s)]
    cnt = Counter(g[0] for g in kg)
    n_cx = cnt.get("cx", 0) + cnt.get("cnot", 0)
    n_cz = cnt.get("cz", 0)
    n_1q = len(kg) - n_cx - n_cz
    N = 2 ** n
    bytes_moved = (n_1q + n_cx) * 2 * N * 16 + n_cz * 2 * (N // 4) * 16
    bytes_useful = n_1q * 2 * N * 16 + n_cx * N * 16 + n_cz * 2 * (N // 4) * 16
    med = lambda k: float(np.median([r[k] for r in rs]))
    tk = med("T_kernel_us") / 1e6
    row = dict(family=fam, n=n, reps=len(rs), gates_in=rs[0]["n_gates_in"], gates_kernel=len(kg),
               n_1q=n_1q, n_cx=n_cx, n_cz=n_cz,
               T_compile_s=med("T_compile_us") / 1e6, T_h2d_s=med("T_h2d_us") / 1e6, T_kernel_s=tk,
               T_d2h_s=med("T_d2h_us") / 1e6, T_post_s=med("T_post_us") / 1e6, T_e2e_s=med("T_e2e_us") / 1e6,
               kernel_frac=tk / (med("T_e2e_us") / 1e6),
               us_per_kernel_gate=tk / len(kg) * 1e6,
               eff_bw_moved_gbps=bytes_moved / tk / 1e9, eff_bw_useful_gbps=bytes_useful / tk / 1e9,
               min_fidelity=min(r["fidelity"] for r in rs if r["fidelity"] is not None))
    rows.append(row)

os.makedirs(f"{WS}/results/processed", exist_ok=True)
import csv
with open(f"{WS}/results/processed/v07_baseline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{'circuit':10s} {'n':>3s} {'g_in':>5s} {'g_krn':>5s} {'T_kernel':>10s} {'T_e2e':>10s} {'krn%':>5s} "
      f"{'us/gate':>10s} {'BW_moved':>9s} {'BW_useful':>9s} {'minF':>8s}")
for r in rows:
    print(f"{r['family']:10s} {r['n']:3d} {r['gates_in']:5d} {r['gates_kernel']:5d} {r['T_kernel_s']:10.4f} "
          f"{r['T_e2e_s']:10.4f} {100 * r['kernel_frac']:5.1f} {r['us_per_kernel_gate']:10.1f} "
          f"{r['eff_bw_moved_gbps']:9.3f} {r['eff_bw_useful_gbps']:9.3f} {1 - r['min_fidelity']:8.1e}")
