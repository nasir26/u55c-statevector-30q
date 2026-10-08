"""Compare the other host's FPGA timing subset with the original host's flagship results (median T_kernel, T_e2e)."""
import glob, json, os, sys
import numpy as np
WS = os.environ.get("WS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ref = []
for f in glob.glob(f"{WS}/results/raw/phase5/sv_main_*_sv_e16w_m4.json"):
    ref += json.load(open(f))["records"]
for f in glob.glob(f"{WS}/results/raw/crosscheck_*/sv_crosscheck_sv_e16w_m4.json"):
    oth = json.load(open(f))["records"]
    print("==", f)
    for fam in ("ghz", "qft", "rqc", "vqe"):
        for n in (20, 24, 28):
            sel = lambda rs: [r for r in rs if r["family"] == fam and r["n"] == n and not r["warmup"] and (fam != "rqc" or r["depth"] == 10)]
            a, b = sel(ref), sel(oth)
            if a and b:
                ka, kb = np.median([r["T_kernel"] for r in a]), np.median([r["T_kernel"] for r in b])
                ea, eb = np.median([r["T_e2e"] for r in a]), np.median([r["T_e2e"] for r in b])
                print(f"{fam:4s} n={n:2d} kernel {ka:8.4f} vs {kb:8.4f} ({100 * (kb / ka - 1):+5.1f}%)  e2e {ea:8.4f} vs {eb:8.4f} ({100 * (eb / ea - 1):+5.1f}%)")
