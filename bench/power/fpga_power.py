"""Background power/thermal sampler for U55C cards (read-only sysfs, XMC sensors).

The XMC refreshes its readings about once per second (measured: 1.005 s, docs/ENERGY.md), so the sampler
polls every 0.1 s and records each distinct reading with a host timestamp. Board power is
12V_PEX*I + 12V_AUX*I + 3V3_PEX*I (equals xbutil's power_consumption_watts).

Safety: if any card reports FPGA >= 90 C or HBM >= 85 C (conservative limits chosen for this campaign),
the sampler writes an ABORT line and sets the abort flag file; benchmark drivers check it between runs.

Usage as a library:
    s = Sampler(["0000:41:00.1"], "results/raw/.../power.csv"); s.start(); ...; s.stop()
    s.mean_power(bdf, t0, t1)   # time-weighted mean board power over [t0, t1] (host perf_counter times)
"""
import glob, os, threading, time, csv

FPGA_LIMIT_C = 90
HBM_LIMIT_C = 85


def xmc_dir(bdf):
    d = glob.glob(f"/sys/bus/pci/devices/{bdf}/xmc.u.*")
    if not d:
        raise RuntimeError(f"no XMC sysfs for {bdf}")
    return d[0]


def _r(d, f):
    with open(os.path.join(d, f)) as fh:
        return int(fh.read().split()[0])


def read(bdf_dir):
    v = {}
    for rail in ("12v_pex", "12v_aux", "3v3_pex"):
        vol = _r(bdf_dir, f"xmc_{rail}_vol") / 1000.0
        cur = _r(bdf_dir, f"xmc_{rail}_curr") / 1000.0
        v[f"{rail}_w"] = vol * cur
    v["board_w"] = v["12v_pex_w"] + v["12v_aux_w"] + v["3v3_pex_w"]
    v["vccint_a"] = _r(bdf_dir, "xmc_0v85_curr") / 1000.0 if os.path.exists(os.path.join(bdf_dir, "xmc_0v85_curr")) else None
    v["fpga_c"] = _r(bdf_dir, "xmc_fpga_temp")
    v["hbm_c"] = _r(bdf_dir, "xmc_hbm_temp")
    return v


def wait_cool(bdf, limit_c=80, timeout_s=900, poll_s=5):
    """Block until the card's FPGA temperature is below limit_c (pacing for heavy runs). Returns seconds waited."""
    d = xmc_dir(bdf); t0 = time.time()
    while _r(d, "xmc_fpga_temp") >= limit_c and time.time() - t0 < timeout_s:
        time.sleep(poll_s)
    return time.time() - t0


class Sampler:
    def __init__(self, bdfs, out_csv, period=0.1, abort_flag=None):
        self.bdfs = list(bdfs)
        self.dirs = {b: xmc_dir(b) for b in self.bdfs}
        self.out_csv = out_csv
        self.period = period
        self.abort_flag = abort_flag or out_csv + ".ABORT"
        self.samples = {b: [] for b in self.bdfs}   # (t, dict) on change
        self._stop = threading.Event()
        self.aborted = False

    def _loop(self):
        last = {b: None for b in self.bdfs}
        with open(self.out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_perf", "t_unix", "bdf", "board_w", "12v_pex_w", "12v_aux_w", "3v3_pex_w", "fpga_c", "hbm_c"])
            while not self._stop.is_set():
                t, tu = time.perf_counter(), time.time()
                for b in self.bdfs:
                    v = read(self.dirs[b])
                    key = (round(v["board_w"], 4), v["fpga_c"], v["hbm_c"])
                    if key != last[b]:
                        last[b] = key
                        self.samples[b].append((t, v))
                        w.writerow([f"{t:.4f}", f"{tu:.4f}", b, f"{v['board_w']:.4f}", f"{v['12v_pex_w']:.4f}",
                                    f"{v['12v_aux_w']:.4f}", f"{v['3v3_pex_w']:.4f}", v["fpga_c"], v["hbm_c"]])
                        fh.flush()
                    if v["fpga_c"] >= FPGA_LIMIT_C or v["hbm_c"] >= HBM_LIMIT_C:
                        self.aborted = True
                        with open(self.abort_flag, "w") as af:
                            af.write(f"ABORT {b} fpga={v['fpga_c']} hbm={v['hbm_c']} t={tu}\n")
                time.sleep(self.period)

    def start(self):
        self._th = threading.Thread(target=self._loop, daemon=True); self._th.start(); return self

    def stop(self):
        self._stop.set(); self._th.join()

    def mean_power(self, bdf, t0, t1):
        """Time-weighted mean of board power over [t0, t1] using the step function of recorded readings."""
        s = self.samples[bdf]
        if not s:
            return None
        acc = 0.0
        for i, (t, v) in enumerate(s):
            ta = max(t, t0)
            tb = min(s[i + 1][0] if i + 1 < len(s) else t1, t1)
            if tb > ta:
                acc += v["board_w"] * (tb - ta)
        first = s[0][0]
        if first > t0:  # extend first reading backwards
            acc += s[0][1]["board_w"] * (min(first, t1) - t0)
        return acc / (t1 - t0)

    def n_readings(self, bdf, t0, t1):
        return sum(1 for t, _ in self.samples[bdf] if t0 <= t <= t1)
