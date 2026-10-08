"""Load every raw result file into one tidy pandas DataFrame with common columns.

Common columns: platform, config, family, n, depth, rep, warmup, T_compile, T_h2d, T_kernel, T_exchange, T_d2h,
T_post, T_e2e (seconds; NaN where a platform has no such stage), fidelity, board_w (device power, W), source.
Platform-specific extras are kept as additional columns.
"""
import glob, json, os
import numpy as np
import pandas as pd

WS = os.environ.get("WS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = f"{WS}/results/raw"
TCOLS = ["T_compile", "T_h2d", "T_kernel", "T_exchange", "T_d2h", "T_post", "T_e2e"]


def _v07():
    rows = []
    for f in glob.glob(f"{RAW}/phase*/v07_baseline_*.json"):
        for r in json.load(open(f))["records"]:
            rows.append(dict(platform="fpga_v07", config="v07 (complex128, 8 banks)", family=r["family"], n=r["n"],
                             depth=10 if r["family"] == "rqc" else None, rep=r["rep"], warmup=False,
                             T_compile=r["T_compile_us"] / 1e6, T_h2d=r["T_h2d_us"] / 1e6, T_kernel=r["T_kernel_us"] / 1e6,
                             T_exchange=0.0, T_d2h=r["T_d2h_us"] / 1e6, T_post=r["T_post_us"] / 1e6,
                             T_e2e=r["T_e2e_us"] / 1e6, fidelity=r["fidelity"], source=os.path.relpath(f, WS)))
    return rows


def _sv():
    rows = []
    for f in [x for pat in ("sv_main_*", "sv_builds_*", "sv_ablate_*", "sv_energy_*") for x in glob.glob(f"{RAW}/phase5/{pat}.json")]:
        d = json.load(open(f))
        for r in d["records"]:
            if r.get("energy_loop") or "xclbin" not in r:
                continue
            cfg = f"{r['xclbin']} P={r['P']} k={r['k']} fuse={r['fuse']} tiling={r['tiling']} {r['layout']}" \
                  f"{'' if r.get('device_swap', True) else ' hostswap'}"
            rows.append(dict(platform="fpga_sv", config=cfg, family=r["family"], n=r["n"], depth=r.get("depth"),
                             rep=r["rep"], warmup=r["warmup"], **{t: r[t] for t in TCOLS}, fidelity=r["fidelity"],
                             board_w=r.get("board_w"), P=r["P"], k=r["k"], engines=r["engines"], tag=r.get("tag"),
                             n_passes=r["stats"]["n_passes"], n_remaps=r["stats"]["n_remaps"],
                             n_ops=r["stats"]["n_ops"], source=os.path.relpath(f, WS)))
    return rows


def _cpu():
    rows = []
    for f in [x for pat in ("cpu_aer_*", "cpu_qsim_*", "cpu_qulacs_*", "cpu_numpy1t*") for x in glob.glob(f"{RAW}/phase5/{pat}.json")]:
        d = json.load(open(f))
        if "records" not in d:
            continue
        for r in d["records"]:
            cfg = r["platform"].replace("cpu_", "") + (f" {r['threads']}t" if r["platform"] != "cpu_numpy1t" else "") + \
                  (f" {r['precision']} fusion={r['fusion']}" if r["platform"] == "cpu_aer" else "") + \
                  (f" {d['args'].get('tag', '')}" if d["args"].get("tag") else "")
            run = r["T_run_s"]; comp = r.get("T_compile_s", 0.0)
            rows.append(dict(platform=r["platform"], config=cfg.strip(), family=r["family"], n=r["n"],
                             depth=10 if r["family"] == "rqc" else None, rep=r["rep"], warmup=r["rep"] == 0,
                             T_compile=comp, T_h2d=0.0, T_kernel=run, T_exchange=0.0, T_d2h=0.0, T_post=np.nan,
                             T_e2e=comp + run, fidelity=r.get("fidelity_vs_qiskit"), source=os.path.relpath(f, WS)))
    return rows


def _gpu():
    rows = []
    for f in glob.glob(f"{RAW}/gpu/*.jsonl"):
        if os.path.basename(f).startswith("smoke"):
            continue
        for line in open(f):
            r = json.loads(line)
            if "meta" in r or r.get("energy_loop") or r.get("error"):
                continue
            cfg = f"{r['platform'].replace('gpu_', '')} {r['precision']}" + \
                  (f" cusv={r['custatevec']}" if r["platform"] == "gpu_aer_gpu" else "") + f" [{os.path.basename(f).split('_')[0]}]"
            rows.append(dict(platform=r["platform"], config=cfg, family=r["family"], n=r["n"],
                             depth=10 if r["family"] == "rqc" else None, rep=r["rep"], warmup=r["warmup"],
                             T_compile=r["T_compile_s"], T_h2d=0.0, T_kernel=r["T_run_s"], T_exchange=0.0,
                             T_d2h=r["T_d2h_s"], T_post=np.nan, T_e2e=r["T_compile_s"] + r["T_run_s"] + r["T_d2h_s"],
                             fidelity=r["fidelity"], board_w=(r.get("power") or {}).get("mean_w"),
                             source=os.path.relpath(f, WS)))
    return rows


def load_runs():
    df = pd.DataFrame(_v07() + _sv() + _cpu() + _gpu())
    return df


def summarise(df, keys=("platform", "config", "family", "n", "depth")):
    d = df[~df["warmup"].fillna(False).astype(bool)]
    g = d.groupby(list(keys), dropna=False)
    out = g[TCOLS].median()
    out["T_e2e_q1"] = g["T_e2e"].quantile(0.25); out["T_e2e_q3"] = g["T_e2e"].quantile(0.75)
    out["T_kernel_q1"] = g["T_kernel"].quantile(0.25); out["T_kernel_q3"] = g["T_kernel"].quantile(0.75)
    out["T_kernel_min"] = g["T_kernel"].min()
    out["reps"] = g.size()
    out["min_fidelity"] = g["fidelity"].min()
    if "board_w" in d:
        out["board_w"] = g["board_w"].median()
    return out.reset_index()


def load_hbm():
    rows = []
    for f in glob.glob(f"{RAW}/phase2/hbm_*.json"):
        d = json.load(open(f))
        for r in d["records"]:
            rows.append(dict(variant=os.path.basename(f).replace(".json", ""), **r, source=os.path.relpath(f, WS)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_runs()
    print(df.groupby(["platform"]).size())
    print(summarise(df).head(20).to_string())
