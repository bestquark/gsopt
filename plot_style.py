from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FONT_LARGE = 18
FONT_MED = 14
FONT_SMALL = 11


def apply_style():
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": FONT_SMALL,
            "axes.titlesize": FONT_LARGE,
            "axes.labelsize": FONT_MED,
            "xtick.labelsize": FONT_SMALL,
            "ytick.labelsize": FONT_SMALL,
            "legend.fontsize": FONT_SMALL,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.12,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "lines.linewidth": 2.2,
            "lines.markersize": 5,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#161616",
            "axes.labelcolor": "#161616",
            "axes.titlepad": 10,
            "legend.frameon": False,
            "legend.handlelength": 2.8,
            "legend.handletextpad": 0.6,
            "legend.columnspacing": 1.0,
        }
    )


def inferno_colors(n: int, start: float = 0.12, stop: float = 0.92):
    cmap = plt.get_cmap("inferno")
    if n <= 1:
        return [cmap(0.65)]
    return [cmap(x) for x in np.linspace(start, stop, n)]


def finish_axes(ax, xlabel: str | None = None, ylabel: str | None = None, title: str | None = None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.tick_params(length=6, width=1.0)


def annotate_panel(ax, text: str):
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SMALL - 1,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#dddddd", "alpha": 0.95},
    )
