"""Architecture block diagram (paper Fig. 1), drawn with matplotlib patches as a vector PDF."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

style.apply()
WS = os.environ.get("WS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{WS}/paper/revised/figures"
BLUE, ORANGE, AQUA, INK2 = "#2a78d6", "#eb6834", "#1baf7a", style.INK2
FILL = {"host": "#e9f1fb", "eng": "#e7f3ee", "hbm": "#f3f2ef", "swap": "#fdeee6"}

fig, ax = plt.subplots(figsize=(7.2, 4.1))
ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")


def box(x, y, w, h, text, fc, ec=INK2, fs=7.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=style.INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(x0, y0, x1, y1, color=INK2, both=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<|-|>" if both else "-|>", mutation_scale=8,
                                 lw=0.9, color=color))


# host
box(2, 44, 31, 11, "Host compiler\nQiskit circuit $\\to$ MUX2 records\n1q fusion, pass/remap scheduling", FILL["host"], BLUE, fs=6.8)
box(36, 44, 31, 11, "Host runtime (XRT)\nlaunch CUs per segment, remaps,\ndevice zeroing, sampling", FILL["host"], BLUE, fs=6.8)
box(70, 44, 27, 11, "numpy model of the\ninstruction stream\n(validation only)", "#f7f7f5", INK2, fs=6.8)
arrow(33, 49.5, 36, 49.5); arrow(67, 49.5, 70, 49.5)
ax.text(51, 40.2, "PCIe Gen3$\\times$16: instructions, final state", ha="center", fontsize=6.8, color=INK2)
arrow(51, 44, 51, 38.5, both=True)

# FPGA region
ax.add_patch(FancyBboxPatch((1.5, 1.5), 97, 36, boxstyle="round,pad=0.3,rounding_size=1.5", fc="none", ec=INK2,
                            lw=0.8, linestyle=(0, (3, 2))))
ax.text(3, 35.3, "Alveo U55C", fontsize=7.5, color=INK2, fontweight="bold")
# one engine in detail
box(4, 18, 52, 15.5, "", FILL["eng"], AQUA)
ax.text(6, 31.3, "Engine CU (one or two pseudo-channels each)", fontsize=7, color=style.INK, fontweight="bold")
box(6, 21, 13, 8.5, "reader\nbursts of\nsub-blocks", "white", AQUA, fs=6.6)
box(22, 21, 15, 8.5, "compute\nMUX2 list on tile\n4 pairs / cycle", "white", AQUA, fs=6.6)
box(40, 21, 13, 8.5, "writer\nbursts of\nsub-blocks", "white", AQUA, fs=6.6)
arrow(19, 25.2, 22, 25.2); arrow(37, 25.2, 40, 25.2)
ax.text(29.5, 18.6, "$2^{12}$-amplitude tile, parity-banked (II = 1)", ha="center", va="bottom", fontsize=6.0, color=INK2)
# more engines
for i in range(3):
    box(58 + i * 3.2, 18 + i * 0.0, 2.6, 15.5, "", FILL["eng"], AQUA)
ax.text(62.3, 16.2, "16 engine CUs", fontsize=6.4, color=style.INK, ha="center")
# swap CUs
box(72, 18, 24, 15.5, "swap CUs (4)\nexchange half-slices\nbetween channels for\nbank-qubit remaps", FILL["swap"], ORANGE, fs=6.6)
# HBM
box(4, 4, 92, 9.5, "", FILL["hbm"], INK2)
for i in range(32):
    x = 5.2 + i * 2.82
    ax.add_patch(FancyBboxPatch((x, 5.4), 2.3, 5.2, boxstyle="round,pad=0.05,rounding_size=0.3", fc="white", ec=INK2, lw=0.5))
ax.text(50, 12.0, "HBM2: 32 pseudo-channels $\\times$ 512 MB (the statevector, one slice per channel)", ha="center", fontsize=6.6, color=INK2)
arrow(12, 18, 12, 13.6, color=AQUA, both=True); arrow(46, 18, 46, 13.6, color=AQUA, both=True)
arrow(84, 18, 84, 13.6, color=ORANGE, both=True)
ax.text(15.5, 15.6, "one AXI port", fontsize=6.2, color=INK2)
ax.text(85.5, 15.7, "all channels\n(HBM switch)", fontsize=6.0, color=INK2, va="center", ha="left")
fig.savefig(f"{OUT}/fig_architecture.pdf"); fig.savefig(f"{OUT}/fig_architecture.png", dpi=200)
print("wrote fig_architecture")
