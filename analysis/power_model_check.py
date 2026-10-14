"""R1.4: validate the submitted manuscript's analytical Alveo power/energy model (fpga_simulator_v07.EnergyEstimator,
with its own cycle and HBM-traffic estimates from _profile_gates) against measured board power (XMC, 1 Hz) for the
same v07 runs. Output: results/processed/power_model_check.csv + printed table."""
import json, os, sys, csv
WS = os.environ.get("WS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))); sys.path[:0] = [f"{WS}/host/v07", f"{WS}/circuits"]
import fpga_simulator_v07 as v07
from generators import make, to_tuples
d = json.load(open(f"{WS}/results/raw/phase5/v07_energy.json"))
idle = [r for r in d["records"] if r.get("kind")][0]["board_w"]
rows = []
for r in d["records"]:
    if r.get("kind"):
        continue
    qc = make(r["family"], r["n"])
    segs, _, _ = v07._validate_and_split_gates(to_tuples(qc))
    kg = [g for s in segs for g in v07._lower_to_kernel_basis(s)]
    gp = v07.QuantumFPGACircuit._profile_gates(None, kg, r["n"])
    cyc = sum(g.est_cycles for g in gp); hbm = sum((g.hbm_read_kb + g.hbm_write_kb) * 1024 for g in gp)
    e_mj, p_w = v07.EnergyEstimator.total_energy_mj(cyc, hbm)
    t_model = cyc / (v07.KERNEL_CLOCK_MHZ * 1e6)
    t_meas = r["per_run_s"] * r["kernel_frac"]                       # measured kernel time per run
    e_meas = r["board_w"] * t_meas                                   # measured board energy over the kernel time
    rows.append(dict(family=r["family"], n=r["n"], model_power_w=p_w, measured_board_w=r["board_w"], idle_w=idle,
                     power_err_pct=100 * (p_w - r["board_w"]) / r["board_w"], model_kernel_s=t_model,
                     measured_kernel_s=t_meas, model_energy_j=e_mj / 1e3, measured_energy_j=e_meas,
                     energy_err_pct=100 * (e_mj / 1e3 - e_meas) / e_meas))
os.makedirs(f"{WS}/results/processed", exist_ok=True)
with open(f"{WS}/results/processed/power_model_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['family']:4s} n={r['n']:2d} model P={r['model_power_w']:5.1f} W measured {r['measured_board_w']:5.1f} W "
          f"(err {r['power_err_pct']:+5.0f}%) | model t={r['model_kernel_s']:8.3f}s meas {r['measured_kernel_s']:8.3f}s | "
          f"model E={r['model_energy_j']:8.2f} J meas {r['measured_energy_j']:8.2f} J (err {r['energy_err_pct']:+5.0f}%)")
