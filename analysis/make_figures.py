"""Generate every figure of the revised manuscript from results/raw (via analysis/load.py).

Outputs vector PDF + PNG into paper/revised/figures/ (or FIG_OUT). Each figure function documents which timing field
(revision Sec. 6) it plots. Run: envs/fpga/bin/python analysis/make_figures.py [names...]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402
from load import load_runs, summarise, WS  # noqa: E402

OUT = os.environ.get("FIG_OUT", f"{WS}/paper/revised/figures")
os.makedirs(OUT, exist_ok=True)
style.apply()
FAM_TITLE = {"ghz": "GHZ", "qft": "QFT", "rqc": "RQC (depth 10)", "vqe": "VQE ansatz (2 layers)"}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", metadata={"CreationDate": None}); fig.savefig(f"{OUT}/{name}.png", dpi=200, metadata={"Software": None}); plt.close(fig)
    print("wrote", name)


def pick(summ, platform, config_sub=None):
    d = summ[summ.platform == platform]
    if config_sub:
        d = d[d.config.str.contains(config_sub, regex=False)]
    return d


def fig_scaling(summ, field="T_kernel", name="fig_scaling_kernel", series=None):
    """Small multiples per family: median `field` vs n (log y), IQR band. One axis per panel."""
    series = series or [
        ("fpga_sv", "fpga_sv", "main", style.LABEL["fpga_sv"]),
        ("fpga_v07", "fpga_v07", None, style.LABEL["fpga_v07"]),
        ("cpu", "cpu_aer", "128t single fusion=1", "CPU, Qiskit Aer, 128 threads, fp32"),
        ("gpu", "gpu_aer_gpu", "single cusv=1 [a100]", "GPU, A100, Aer + cuStateVec, fp32"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    for ax, fam in zip(axs.flat, ["ghz", "qft", "rqc", "vqe"]):
        for key, plat, cfg, lab in series:
            d = pick(summ, plat, cfg)
            d = d[(d.family == fam) & ((d.depth.isna()) | (d.depth == 10))].sort_values("n")
            if d.empty:
                continue
            st = style.series_style(key)
            ax.plot(d.n, d[field], label=lab, **st)
            q1, q3 = f"{field}_q1", f"{field}_q3"
            if q1 in d:
                ax.fill_between(d.n, d[q1], d[q3], color=st["color"], alpha=0.15, linewidth=0)
        ax.set_yscale("log"); ax.set_title(FAM_TITLE[fam], loc="left")
        ax.set_xlabel("qubits n"); ax.set_ylabel(f"{field.replace('_', ' ')} per circuit [s]")
    h, l = axs[0, 0].get_legend_handles_labels()
    for ax in axs.flat:
        if len(ax.get_legend_handles_labels()[0]) > len(h):
            h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, name)


# ------------------------------------------------------------------ flagship time breakdown
def fig_breakdown(summ=None):
    """Stacked median T_compile, T_h2d, T_kernel, T_exch, T_d2h, T_post for the flagship build, n = 24..30."""
    import json
    recs = []
    for f in ("sv_main_small_sv_e16w_m4.json", "sv_main_large_sv_e16w_m4.json"):
        p = f"{WS}/results/raw/phase5/{f}"
        if os.path.exists(p):
            recs += json.load(open(p))["records"]
    fields = [("T_compile", "compile", "#c3c2b7"), ("T_h2d", "host-to-device", "#9ec5f4"), ("T_kernel", "kernel", "#2a78d6"),
              ("T_exchange", "exchange (remaps)", "#eb6834"), ("T_d2h", "device-to-host", "#8fd6bb"), ("T_post", "sampling", "#52514e")]
    ns = [24, 26, 28, 29, 30]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=False)
    for ax, fam in zip(axs, ["ghz", "qft", "rqc", "vqe"]):
        bottoms = np.zeros(len(ns))
        for key, lab, col in fields:
            vals = []
            for n in ns:
                rs = [r for r in recs if r["family"] == fam and r["n"] == n and not r["warmup"] and (fam != "rqc" or r["depth"] == 10)]
                vals.append(np.median([r[key] for r in rs]) if rs else 0.0)
            ax.bar([str(n) for n in ns], vals, bottom=bottoms, color=col, label=lab, width=0.72, edgecolor=style.SURFACE, linewidth=0.6)
            bottoms += np.array(vals)
        ax.set_title(FAM_TITLE[fam], loc="left"); ax.set_xlabel("qubits n")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("median time per circuit [s]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1), fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig_breakdown")


# ------------------------------------------------------------------ CU scaling (e8, n = 24)
def fig_cu_scaling(summ=None):
    import json
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    marks = {"ghz": "o", "qft": "s", "rqc": "^", "vqe": "D"}
    cols = {"ghz": "#2a78d6", "qft": "#eb6834", "rqc": "#1baf7a", "vqe": "#4a3aa7"}
    for fam in ("ghz", "qft", "rqc", "vqe"):
        xs, ys = [], []
        for pm in (1, 2, 4, 8):
            p = f"{WS}/results/raw/phase5/sv_ablate_e8_pmax{pm}_n24.json"
            if not os.path.exists(p):
                continue
            rs = [r for r in json.load(open(p))["records"] if r["family"] == fam and not r["warmup"]]
            xs.append(pm); ys.append(np.median([r["T_kernel"] for r in rs]))
        ax.plot(xs, ys, marker=marks[fam], color=cols[fam], label=FAM_TITLE[fam].split(" (")[0],
                markeredgecolor=style.SURFACE, markeredgewidth=0.8)
        if xs:
            ax.plot(xs, [ys[0] / x for x in xs], color=cols[fam], linestyle=":", linewidth=0.9)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("active engine CUs (= pseudo-channels)"); ax.set_ylabel("$T_\\mathrm{kernel}$ at $n=24$ [s]")
    ax.legend(title="dotted: ideal 1/P", title_fontsize=6.5)
    fig.tight_layout(); save(fig, "fig_cu_scaling")


# ------------------------------------------------------------------ roofline (flagship)
def fig_roofline(summ=None):
    """Achieved GFLOP/s of the flagship kernel vs arithmetic intensity of its passes (28 flop per pair update,
    32 bytes per pair per pass); roofs: measured 16-channel bandwidth and the engines' compute peak."""
    import json, re
    macros = dict(re.findall(r"\\newcommand\{\\Num(\w+)\}\{([^}]*)\}", open(f"{WS}/paper/generated/macros.tex").read()))
    bw = float(macros["HbmSixteenCopy"]); bw32 = float(macros["HbmThirtytwoCopy"]); peak = float(macros["FlagComputeRoof"])
    recs = []
    for f in ("sv_main_small_sv_e16w_m4.json", "sv_main_large_sv_e16w_m4.json"):
        recs += json.load(open(f"{WS}/results/raw/phase5/{f}"))["records"]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = np.logspace(-1, 1.5, 200)
    ax.plot(x, np.minimum(peak, bw * x), color=style.INK2, linewidth=1.2)
    ax.plot(x, np.minimum(peak, bw32 * x), color=style.INK2, linewidth=0.8, linestyle="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.08, 40); ax.set_ylim(5, peak * 2)
    fig.canvas.draw()
    def label_on(slope_bw, x0, text):
        p0 = ax.transData.transform((x0, slope_bw * x0)); p1 = ax.transData.transform((x0 * 2, slope_bw * x0 * 2))
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(x0, slope_bw * x0 * 1.18, text, fontsize=6.2, color=style.INK2, rotation=ang, rotation_mode="anchor")
    label_on(bw, 0.11, f"16 channels, {bw:.0f} GB/s (measured)")
    label_on(bw32, 0.11, f"32 channels, {bw32:.0f} GB/s (measured)")
    ax.text(10, peak * 1.08, f"engine compute peak {peak:.0f} GFLOP/s", fontsize=6.3, color=style.INK2, ha="center")
    cols = {"ghz": "#2a78d6", "qft": "#eb6834", "rqc": "#1baf7a", "vqe": "#4a3aa7"}
    marks = {"ghz": "o", "qft": "s", "rqc": "^", "vqe": "D"}
    for fam in ("ghz", "qft", "rqc", "vqe"):
        pts = []
        for n in (26, 28, 29, 30):
            rs = [r for r in recs if r["family"] == fam and r["n"] == n and not r["warmup"] and (fam != "rqc" or r["depth"] == 10)]
            if not rs:
                continue
            st = rs[0]["stats"]; tk = np.median([r["T_kernel"] for r in rs])
            ai = FLOP / BYTES * st["n_ops"] / st["n_passes"]
            gf = st["n_ops"] * (2 ** (n - 1)) * FLOP / tk / 1e9
            pts.append((ai, gf))
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=marks[fam], color=cols[fam], s=22,
                   edgecolor=style.SURFACE, linewidth=0.6, label=FAM_TITLE[fam].split(" (")[0], zorder=3)
    ax.set_xlabel("arithmetic intensity [flop/byte]"); ax.set_ylabel("achieved [GFLOP/s]")
    ax.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout(); save(fig, "fig_roofline")


# ------------------------------------------------------------------ cross-platform T_e2e (series chosen by source file)
GPU_HOST = os.environ.get("GPU_HOST_TAG", "a100")


def fig_xplat(summ=None, field="T_e2e"):
    """Small multiples per family: median T_e2e vs n for the new FPGA kernel (flagship), the v07 baseline,
    CPU Aer fp32 and qsim fp32 (128 threads) and, when measured, GPU Aer fp32 with cuStateVec."""
    df = load_runs()
    df = df[(df.depth.isna()) | (df.depth == 10)]
    series = [
        ("fpga_sv", df.source.str.contains(r"sv_main_.*sv_e16w_m4", regex=True), "FPGA, new kernel (U55C, fp32)"),
        ("cpu", (df.platform == "cpu_aer") & (df.config == "aer 128t single fusion=1"), "CPU, Qiskit Aer fp32, 128 cores"),
        ("cpu_alt", (df.platform == "cpu_qsim"), "CPU, qsim fp32, 128 cores"),
        ("gpu", (df.platform == "gpu_aer_gpu") & df.config.str.contains(f"single cusv=1 [{GPU_HOST}]", regex=False),
         "GPU, Aer fp32 + cuStateVec (A100)"),
        ("fpga_v07", df.platform == "fpga_v07", "FPGA, v07 baseline (fp64)"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    for ax, fam in zip(axs.flat, ["ghz", "qft", "rqc", "vqe"]):
        for key, mask, lab in series:
            d = summarise(df[mask])
            d = d[d.family == fam].sort_values("n")
            if d.empty:
                continue
            st = style.series_style(key)
            ax.plot(d.n, d[field], label=lab, **st)
            ax.fill_between(d.n, d[f"{field}_q1"], d[f"{field}_q3"], color=st["color"], alpha=0.15, linewidth=0)
        ax.set_yscale("log"); ax.set_title(FAM_TITLE[fam], loc="left")
        ax.set_ylabel("$T_\\mathrm{e2e}$ per circuit [s]")
        if ax in axs[1]:
            ax.set_xlabel("qubits n")
    h, l = [], []
    for ax in axs.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "fig_xplat")


# ------------------------------------------------------------------ latency distribution (2000 runs per circuit)
def fig_jitter(summ=None):
    """ECDF of T_e2e (top) and T_kernel (bottom) at n = 16 for each platform that recorded it."""
    import json
    files = [("fpga_sv", "jitter_sv_sv_e16w_m4.json", "FPGA, new kernel"), ("fpga_v07", "jitter_v07.json", "FPGA, v07"),
             ("cpu", "jitter_cpu_aer8.json", "CPU, Aer fp32, 8 threads"), ("cpu_alt", "jitter_cpu_qulacs.json", "CPU, Qulacs fp64")]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.4))
    for j, fam in enumerate(("ghz", "rqc")):
        for i, fld in enumerate(("T_e2e", "T_kernel")):
            ax = axs[i, j]
            for key, f, lab in files:
                p = f"{WS}/results/raw/phase5/{f}"
                if not os.path.exists(p):
                    continue
                rs = [r for r in json.load(open(p))["records"] if r["family"] == fam and r["n"] == 16 and r.get(fld)]
                if not rs:
                    continue
                x = np.sort(np.array(rs[0][fld][rs[0].get("warmup", 0):]) * 1e3)
                st = style.series_style(key)
                ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=st["color"], linestyle=st["linestyle"], label=lab)
            lo, hi = (min(l.get_xdata().min() for l in ax.lines), max(l.get_xdata().max() for l in ax.lines)) if ax.lines else (1, 1)
            if hi / lo > 10:
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_ylim(0, 1.01)
            ax.set_title(f"{FAM_TITLE[fam].split(' (')[0]}-16, {'end to end' if fld == 'T_e2e' else 'kernel / simulator run'}", loc="left")
            ax.set_xlabel("time per run [ms]"); ax.set_ylabel("fraction of runs")
    h, l = [], []
    for ax in axs.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save(fig, "fig_jitter")


# ------------------------------------------------------------------ builds (kernel time per build, n = 20..30)
def fig_builds(summ=None):
    import json
    builds = [("sv_e8", "8 engines", "#8a8985", "x", ":"), ("sv_e16", "16 engines", "#eb6834", "s", "--"),
              ("sv_e16w", "16 engines, 2 channels each", "#4a3aa7", "D", "-."),
              ("sv_e16w512", "as before, 512-bit ports", "#1baf7a", "^", "--"),
              ("sv_e16w_m4", "as before, 4 high qubits (main)", "#2a78d6", "o", "-")]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=False)
    for ax, fam in zip(axs, ["ghz", "qft", "rqc", "vqe"]):
        for b, lab, col, mk, ls in builds:
            p = f"{WS}/results/raw/phase5/sv_builds_{b}.json"
            if not os.path.exists(p):
                continue
            recs = [r for r in json.load(open(p))["records"] if r["family"] == fam and not r["warmup"] and not r.get("energy_loop")
                    and (fam != "rqc" or r["depth"] == 10)]
            ns = sorted({r["n"] for r in recs})
            ys = [np.median([r["T_kernel"] for r in recs if r["n"] == n]) for n in ns]
            ax.plot(ns, ys, color=col, marker=mk, linestyle=ls, label=lab, markersize=4,
                    markerfacecolor=col if mk != "x" else "none", markeredgecolor=style.SURFACE if mk != "x" else col)
        ax.set_yscale("log"); ax.set_title(FAM_TITLE[fam].split(" (")[0], loc="left"); ax.set_xlabel("qubits n")
    axs[0].set_ylabel("$T_\\mathrm{kernel}$ [s]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=6.6)
    fig.tight_layout(); save(fig, "fig_builds")


FLOP, BYTES = 28, 32


def fig_maxwidth(summ=None):
    """Largest-width verification on one card: T_e2e (left) and infidelity against a complex128 Aer reference
    (right) for every family at n = 24..30 (results/raw/phase5/sv_maxwidth_sv_e16w_m4.json)."""
    import json
    p = f"{WS}/results/raw/phase5/sv_maxwidth_sv_e16w_m4.json"
    if not os.path.exists(p):
        return
    rs = [r for r in json.load(open(p))["records"] if not r["warmup"] and "one_minus_f" in r]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    marks = {"ghz": ("o", "-"), "qft": ("s", "--"), "rqc": ("^", "-."), "vqe": ("D", ":")}
    cols = {"ghz": "#2a78d6", "qft": "#eb6834", "rqc": "#1baf7a", "vqe": "#4a3aa7"}
    floor = 1e-34
    for fam in ("ghz", "qft", "rqc", "vqe"):
        d = sorted([r for r in rs if r["family"] == fam], key=lambda r: r["n"])
        if not d:
            continue
        ns = [r["n"] for r in d]
        m, ls = marks[fam]
        axs[0].plot(ns, [r["T_e2e"] for r in d], marker=m, linestyle=ls, color=cols[fam], label=FAM_TITLE[fam].split(" (")[0])
        axs[1].plot(ns, [max(r["one_minus_f"], floor) for r in d], marker=m, linestyle=ls, color=cols[fam],
                    label=FAM_TITLE[fam].split(" (")[0])
    axs[0].set_yscale("log"); axs[0].set_ylabel("$T_\\mathrm{e2e}$ [s]"); axs[0].set_xlabel("qubits n")
    axs[0].set_title("time, one card", loc="left")
    axs[1].set_yscale("log"); axs[1].set_ylabel("$1-F$ vs complex128 reference"); axs[1].set_xlabel("qubits n")
    axs[1].set_title("accuracy, one card", loc="left")
    for ax in axs:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axs[1].legend(loc="center right", fontsize=7)
    fig.tight_layout(); save(fig, "fig_maxwidth")



FIGS = {"maxwidth": fig_maxwidth, "xplat": fig_xplat, "jitter": fig_jitter, "builds": fig_builds, "breakdown": fig_breakdown, "cu": fig_cu_scaling, "roofline": fig_roofline}

if __name__ == "__main__":
    df = load_runs()
    summ = summarise(df)
    summ.to_csv(f"{WS}/results/processed/summary_runs.csv", index=False)
    names = sys.argv[1:] or list(FIGS)
    for nm in names:
        FIGS[nm](summ)
