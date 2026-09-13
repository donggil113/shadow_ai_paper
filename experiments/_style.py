"""Shared figure style: a validated categorical palette + print-safe encoding.

Colours are the reference categorical palette (slots assigned in fixed order,
never cycled). Every multi-series figure also carries a second encoding --
line style or marker shape -- so identity survives greyscale printing and
colour-vision deficiency, and every series is either direct-labelled or in the
legend. Ordered sweeps over Gamma use a single-hue sequential ramp (light to
dark), never categorical hues.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# categorical slots, fixed order
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948",
}
SLOTS = [C["blue"], C["orange"], C["aqua"], C["violet"], C["red"], C["green"]]
DASHES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]
MARKERS = ["o", "s", "^", "D", "v", "P"]

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e3e2dd"
SURFACE = "#ffffff"


def sequential(n: int, light: float = 0.82, dark: float = 0.18):
    """Single-hue (blue) sequential ramp, light -> dark, for ordered sweeps.

    Tints the base hue toward white by a fraction that decreases from ``light``
    to ``dark``; the darkest step is close to the categorical blue itself.
    """
    import matplotlib.colors as mcolors
    base = np.array(mcolors.to_rgb(C["blue"]))
    fracs = np.linspace(light, dark, max(n, 1))
    return [tuple(np.clip(1.0 - (1.0 - base) * (1.15 - f), 0, 1)) for f in fracs]


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 7,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8,
        "legend.handlelength": 1.8, "legend.columnspacing": 1.2,
        "lines.linewidth": 1.6, "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
    })


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def label_end(ax, x, y, text, color, dx=0.01, **kw):
    """Direct label at the end of a series, in ink -- never in the series colour."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", fontsize=7.5, color=INK2, **kw)
