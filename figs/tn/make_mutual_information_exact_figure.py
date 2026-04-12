from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style
from examples.tn.model_registry import MUTUAL_INFO_MODELS
from examples.tn.reference_energies import reference_mutual_information

FIG_DIR = Path(os.environ.get("AUTORESEARCH_TN_FIG_DIR", Path(__file__).resolve().parent))
OUTPUT_PDF = FIG_DIR / "tn_mutual_information_exact.pdf"
OUTPUT_PNG = FIG_DIR / "tn_mutual_information_exact.png"
PLOT_ORDER = list(MUTUAL_INFO_MODELS)
DISPLAY_TITLES = {
    "heisenberg_xxx_64": r"Heisenberg XXX",
    "xxz_gapless_64": r"Gapless XXZ",
    "tfim_critical_64": r"Critical TFIM",
    "xx_critical_64": r"Critical XX",
}


def configure_style():
    apply_style()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
            "font.family": "serif",
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
        }
    )


def _tick_positions(nsites: int) -> tuple[list[int], list[str]]:
    indices = [0, nsites // 2, nsites - 1]
    deduped: list[int] = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    return deduped, [str(index + 1) for index in deduped]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    matrices: dict[str, np.ndarray] = {}
    all_positive_values: list[float] = []
    for model in PLOT_ORDER:
        matrix = reference_mutual_information(model)
        if matrix is None:
            raise RuntimeError(f"missing reference mutual-information matrix for: {model}")
        arr = np.asarray(matrix, dtype=np.float64)
        np.fill_diagonal(arr, np.nan)
        matrices[model] = arr
        values = arr[np.isfinite(arr) & (arr > 0.0)]
        all_positive_values.extend(values.tolist())

    if not all_positive_values:
        raise RuntimeError("no positive mutual-information entries were found")

    vmin = max(min(all_positive_values), 1e-6)
    vmax = max(all_positive_values)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d9d9d9")
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, len(PLOT_ORDER), figsize=(18.6, 4.9), sharex=True, sharey=True)
    if len(PLOT_ORDER) == 1:
        axes = np.asarray([axes], dtype=object)

    image = None
    for col_index, model in enumerate(PLOT_ORDER):
        ax = axes[col_index]
        matrix = matrices[model]
        image = ax.imshow(matrix, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        nsites = matrix.shape[0]
        ticks, ticklabels = _tick_positions(nsites)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
        ax.set_title(DISPLAY_TITLES[model], pad=10)
        ax.set_xlabel(r"Site $j$")
        if col_index == 0:
            ax.set_ylabel(r"Site $i$")
        else:
            ax.tick_params(labelleft=False)

    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.20, top=0.86, wspace=0.18)
    cbar = fig.colorbar(image, ax=axes, fraction=0.045, pad=0.03)
    cbar.set_label(r"$I_{ij}^{\mathrm{ref}}$", labelpad=14)
    cbar.ax.tick_params(labelsize=24)

    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        json.dumps(
            {
                "reference_pdf": str(OUTPUT_PDF),
                "reference_png": str(OUTPUT_PNG),
                "models": PLOT_ORDER,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
