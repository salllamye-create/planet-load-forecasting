"""Shared figure style for the paper.

Palette validated with the dataviz colour checker (light surface, categorical):
all six slots pass the lightness band, chroma floor, CVD separation
(worst adjacent dE 10.0 deutan), normal-vision floor and 3:1 contrast checks.
Every series also carries a distinct dash pattern and marker so the figures stay
legible in the greyscale print edition.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE two-column geometry
COL, DBL = 3.45, 7.16

PALETTE = ["#0B5FA5", "#C44601", "#00916B", "#7D3C98", "#A6761D", "#3B7DD8"]
INK, MUTED, GRID = "#1F1F1F", "#5A5A5A", "#D8D8D8"
DASHES = [(None, None), (5, 1.6), (1.6, 1.4), (6, 1.6, 1.4, 1.6), (3, 1.3, 1.3, 1.3), (8, 2)]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# fixed, never-cycled assignment: identity -> colour
MODEL_STYLE = {
    "PLANET":        dict(c="#C44601", ls="-",  lw=1.9, m="o", z=10),
    "PatchTST":      dict(c="#0B5FA5", ls="--", lw=1.2, m="s", z=5),
    "iTransformer":  dict(c="#00916B", ls="-.", lw=1.2, m="^", z=4),
    "CalTST":        dict(c="#0B5FA5", ls=(0, (3, 1, 1, 1)), lw=1.3, m="*", z=6),
    "DLinear":       dict(c="#7D3C98", ls=":",  lw=1.2, m="D", z=3),
    "TimeMixer":     dict(c="#A6761D", ls="--", lw=1.1, m="v", z=3),
    "Autoformer":    dict(c="#3B7DD8", ls="-.", lw=1.1, m="P", z=2),
    "Informer":      dict(c="#3B7DD8", ls=":",  lw=1.1, m="X", z=2),
    "NBEATS":        dict(c="#A6761D", ls="-.", lw=1.1, m="*", z=2),
    "FITS":          dict(c="#7D3C98", ls="--", lw=1.1, m="<", z=2),
    "RLinear":       dict(c="#7D3C98", ls="-.", lw=1.0, m=">", z=2),
    "NLinear":       dict(c="#7D3C98", ls=(0, (4, 2, 1, 2)), lw=1.0, m="h", z=2),
    "LSTM":          dict(c="#00916B", ls=":",  lw=1.0, m="p", z=2),
    "TCN":           dict(c="#00916B", ls="--", lw=1.0, m="8", z=2),
    "HistGBR":       dict(c="#A6761D", ls=":",  lw=1.1, m="d", z=2),
    "SeasonalNaive": dict(c=MUTED,     ls=(0, (1, 2)), lw=1.0, m=None, z=1),
    "TSO":           dict(c="#2B2B2B", ls=(0, (6, 2)), lw=1.3, m=None, z=6),
    "Actual":        dict(c="#2B2B2B", ls="-",  lw=1.4, m=None, z=8),
}


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        # IEEE will not accept Type 3 fonts, which is what matplotlib emits by default;
        # 42 selects TrueType and keeps the figure text selectable and searchable.
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.4, "axes.titlesize": 7.8, "axes.labelsize": 7.6,
        "xtick.labelsize": 6.9, "ytick.labelsize": 6.9, "legend.fontsize": 6.7,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.45,
        "grid.alpha": 0.9, "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 2.3,
        "legend.columnspacing": 1.1, "legend.handletextpad": 0.5,
        "lines.solid_capstyle": "round", "lines.antialiased": True,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def style(name):
    d = MODEL_STYLE.get(name, dict(c=MUTED, ls="-", lw=1.0, m=None, z=1))
    return dict(color=d["c"], linestyle=d["ls"], linewidth=d["lw"], zorder=d["z"])


def save(fig, name, outdir=None):
    import os
    outdir = outdir or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "figures")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  [fig] {name}.pdf")
