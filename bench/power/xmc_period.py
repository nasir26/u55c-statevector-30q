"""Measure the refresh interval of the Alveo XMC power sensors (sysfs) on a card under load: poll every 5 ms for
--seconds and record the times at which the board-power reading changes. Read-only."""
import argparse, json, time, numpy as np
from fpga_power import xmc_dir, read
ap = argparse.ArgumentParser(); ap.add_argument("--bdf", required=True); ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--out", required=True); a = ap.parse_args()
d = xmc_dir(a.bdf); last = None; changes = []; t0 = time.perf_counter()
while time.perf_counter() - t0 < a.seconds:
    v = round(read(d)["board_w"], 5)
    if v != last:
        changes.append((time.perf_counter() - t0, v)); last = v
    time.sleep(0.005)
iv = np.diff([c[0] for c in changes[1:]])
json.dump(dict(bdf=a.bdf, seconds=a.seconds, n_changes=len(changes), intervals_s=iv.tolist(),
               median_interval_s=float(np.median(iv)) if len(iv) else None, changes=changes), open(a.out, "w"), indent=1)
print("changes", len(changes), "median interval", np.median(iv) if len(iv) else None)
