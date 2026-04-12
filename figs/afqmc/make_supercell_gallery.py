from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style

from examples.afqmc.model_registry import ACTIVE_SYSTEMS, SYSTEM_SPECS

FIG_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = FIG_DIR / "afqmc_supercell_gallery.pdf"
OUTPUT_PNG = FIG_DIR / "afqmc_supercell_gallery.png"
BOHR_TO_ANGSTROM = 0.529177210903

ATOM_COLORS = {"H": "#d9d2c5", "Li": "#8a63d2", "C": "#2d3642"}
UNIFORM_ATOM_RADIUS = 0.20
SYSTEM_ATOM_RADII = {
    "diamond_prim": 0.135,
}
CENTRAL_FACE_COLOR = "#8f9aac"
GHOST_FACE_COLOR = "#dde5ee"
CENTRAL_EDGE_COLOR = "#5f6978"
GHOST_EDGE_COLOR = "#adb7c3"

VIEW_ANGLES = {
    "h8_cube_pbc": (20, 36),
    "h10_chain_pbc": (14, 28),
    "lih_cubic_pbc": (18, 30),
    "diamond_prim": (20, 28),
}

TITLE_MAP = {
    "h8_cube_pbc": r"$\mathrm{H_8}$ Cube Supercell",
    "h10_chain_pbc": r"$\mathrm{H_{10}}$ Chain Supercell",
    "lih_cubic_pbc": r"$\mathrm{LiH}$ Cubic Supercell",
    "diamond_prim": r"Diamond Primitive Cell $\left(\mathrm{C_2}\right)$",
}

COVALENT_RADII = {
    "H": 0.31,
    "Li": 1.28,
    "C": 0.76,
}


def configure_style() -> None:
    apply_style()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
            "font.family": "serif",
            "font.size": 20,
            "axes.titlesize": 22,
            "axes.titleweight": "regular",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def parse_atoms(atom_spec: str) -> tuple[list[str], np.ndarray]:
    symbols: list[str] = []
    coords: list[list[float]] = []
    for raw_entry in atom_spec.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        symbol, x_str, y_str, z_str = entry.split()
        symbols.append(symbol)
        coords.append([float(x_str), float(y_str), float(z_str)])
    return symbols, np.asarray(coords, dtype=float) * BOHR_TO_ANGSTROM


def lattice_corners(origin: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    a_vec, b_vec, c_vec = lattice
    return np.asarray(
        [
            origin,
            origin + a_vec,
            origin + b_vec,
            origin + c_vec,
            origin + a_vec + b_vec,
            origin + a_vec + c_vec,
            origin + b_vec + c_vec,
            origin + a_vec + b_vec + c_vec,
        ]
    )


def edge_segments(origin: np.ndarray, lattice: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    corners = lattice_corners(origin, lattice)
    edge_indices = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    return [(corners[i], corners[j]) for i, j in edge_indices]


def face_polygons(origin: np.ndarray, lattice: np.ndarray) -> list[np.ndarray]:
    corners = lattice_corners(origin, lattice)
    return [
        corners[[0, 1, 4, 2]],
        corners[[3, 5, 7, 6]],
        corners[[0, 1, 5, 3]],
        corners[[2, 4, 7, 6]],
        corners[[0, 2, 6, 3]],
        corners[[1, 4, 7, 5]],
    ]


def draw_sphere(ax, center: np.ndarray, radius: float, color: str, *, alpha: float = 1.0, resolution: int = 26) -> None:
    u_vals = np.linspace(0.0, 2.0 * math.pi, resolution)
    v_vals = np.linspace(0.0, math.pi, max(12, int(0.72 * resolution)))
    x_vals = center[0] + radius * np.outer(np.cos(u_vals), np.sin(v_vals))
    y_vals = center[1] + radius * np.outer(np.sin(u_vals), np.sin(v_vals))
    z_vals = center[2] + radius * np.outer(np.ones_like(u_vals), np.cos(v_vals))
    surface = ax.plot_surface(
        x_vals,
        y_vals,
        z_vals,
        color=color,
        linewidth=0.0,
        antialiased=True,
        shade=True,
        alpha=alpha,
    )
    surface.set_rasterized(True)


def should_draw_bond(symbol_a: str, symbol_b: str, distance: float) -> bool:
    radius_a = COVALENT_RADII.get(symbol_a, 0.7)
    radius_b = COVALENT_RADII.get(symbol_b, 0.7)
    return distance <= 1.28 * (radius_a + radius_b)


def draw_bonds(ax, symbols: list[str], coords: np.ndarray) -> None:
    for idx in range(len(symbols)):
        for jdx in range(idx + 1, len(symbols)):
            distance = float(np.linalg.norm(coords[idx] - coords[jdx]))
            if not should_draw_bond(symbols[idx], symbols[jdx], distance):
                continue
            ax.plot(
                [coords[idx, 0], coords[jdx, 0]],
                [coords[idx, 1], coords[jdx, 1]],
                [coords[idx, 2], coords[jdx, 2]],
                color="#4f5b66",
                linewidth=3.0,
                solid_capstyle="round",
                alpha=0.92,
                zorder=2,
            )


def draw_cell(ax, origin: np.ndarray, lattice: np.ndarray, *, face_color: str, edge_color: str, face_alpha: float, edge_alpha: float, edge_width: float) -> None:
    collection = Poly3DCollection(
        face_polygons(origin, lattice),
        facecolor=face_color,
        edgecolor="none",
        linewidths=0.0,
        alpha=face_alpha,
    )
    collection.set_rasterized(True)
    ax.add_collection3d(collection)
    for start, end in edge_segments(origin, lattice):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=edge_color,
            linewidth=edge_width,
            alpha=edge_alpha,
            solid_capstyle="round",
            zorder=1,
        )


def translation_vectors(system_name: str, lattice: np.ndarray) -> list[np.ndarray]:
    a_vec, b_vec, c_vec = lattice
    if system_name == "h10_chain_pbc":
        offsets = [np.zeros(3), c_vec, -c_vec]
    else:
        offsets = [np.zeros(3), a_vec, -a_vec, b_vec, -b_vec, c_vec, -c_vec]
    return offsets


def style_axis(ax, min_corner: np.ndarray, max_corner: np.ndarray) -> None:
    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
    ax.grid(False)
    spans = max_corner - min_corner
    max_span = float(max(spans.max(), 1.0))
    center = 0.5 * (min_corner + max_corner)
    half = 0.48 * max_span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("persp", focal_length=0.94)


def render_system(ax, system_name: str) -> None:
    spec = SYSTEM_SPECS[system_name]
    symbols, coords = parse_atoms(spec.atom)
    lattice = np.asarray(spec.lattice_vectors, dtype=float) * BOHR_TO_ANGSTROM
    origin = np.zeros(3, dtype=float)
    offsets = translation_vectors(system_name, lattice)
    atom_radius = SYSTEM_ATOM_RADII.get(system_name, UNIFORM_ATOM_RADIUS)
    ghost_points: list[np.ndarray] = []
    for translation in offsets[1:]:
        draw_cell(
            ax,
            origin + translation,
            lattice,
            face_color=GHOST_FACE_COLOR,
            edge_color=GHOST_EDGE_COLOR,
            face_alpha=0.032,
            edge_alpha=0.16,
            edge_width=1.15,
        )
        shifted = coords + translation
        ghost_points.append(shifted)
        for symbol, center in zip(symbols, shifted):
            draw_sphere(
                ax,
                center,
                atom_radius,
                ATOM_COLORS.get(symbol, "#bbbbbb"),
                alpha=0.18,
                resolution=18,
            )
    draw_cell(
        ax,
        origin,
        lattice,
        face_color=CENTRAL_FACE_COLOR,
        edge_color=CENTRAL_EDGE_COLOR,
        face_alpha=0.072,
        edge_alpha=0.72,
        edge_width=1.7,
    )
    draw_bonds(ax, symbols, coords)
    for symbol, center in zip(symbols, coords):
        draw_sphere(ax, center, atom_radius, ATOM_COLORS.get(symbol, "#bbbbbb"), alpha=1.0, resolution=28)
    corners = lattice_corners(origin, lattice)
    points = [corners, coords]
    for shifted in ghost_points:
        points.append(shifted)
    all_points = np.vstack(points)
    margin = 0.36
    min_corner = all_points.min(axis=0) - margin
    max_corner = all_points.max(axis=0) + margin
    style_axis(ax, min_corner, max_corner)
    elev, azim = VIEW_ANGLES[system_name]
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(TITLE_MAP.get(system_name, spec.label), pad=18)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    fig, axes = plt.subplots(
        1,
        len(ACTIVE_SYSTEMS),
        figsize=(20.5, 5.4),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    for ax, system_name in zip(np.atleast_1d(axes), ACTIVE_SYSTEMS):
        render_system(ax, system_name)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"pdf": str(OUTPUT_PDF), "png": str(OUTPUT_PNG)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
