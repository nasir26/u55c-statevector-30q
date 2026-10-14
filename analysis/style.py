"""Shared figure style for the revision (print / PDF, light surface).

Palette: validated categorical slots (dataviz reference palette; validator run 2026-10-06, all hard gates pass,
aqua below 3:1 contrast -> every series also has a distinct marker + line style and a legend; <= 4 series are
direct-labelled). Identity follows the platform, never its rank: the same platform keeps its color in every figure.
"""
import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BASELINE_GRAY = "#8a8985"      # v07 baseline: recessive neutral, not a categorical slot

PLATFORM = {
    #               color      marker  linestyle
    "fpga_sv":     ("#2a78d6", "o", "-"),     # slot 1 blue   - new FPGA kernel
    "cpu":         ("#eb6834", "s", "-"),     # slot 2 orange - CPU baselines (Aer)
    "gpu":         ("#1baf7a", "^", "-"),     # slot 3 aqua   - GPU baselines
    "cpu_alt":     ("#4a3aa7", "D", "--"),    # slot 7 violet - second CPU simulator
    "fpga_v07":    (BASELINE_GRAY, "x", ":"),  # baseline
}
LABEL = {"fpga_sv": "FPGA, new kernel (measured)", "fpga_v07": "FPGA, v07 baseline (measured)",
         "cpu": "CPU, Qiskit Aer 128 threads (measured)", "cpu_alt": "CPU, second simulator (measured)",
         "gpu": "GPU, A100 (measured)"}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 5.5,
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
        "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def series_style(key):
    c, m, ls = PLATFORM[key]
    return dict(color=c, marker=m, linestyle=ls, markerfacecolor=c if m not in ("x",) else "none",
                markeredgecolor=SURFACE if m not in ("x",) else c, markeredgewidth=0.8 if m != "x" else 1.2)
