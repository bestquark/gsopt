from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyscf import gto, scf
from skimage import measure

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_style import apply_style

FIG_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = FIG_DIR / "vqe_active_orbital_gallery.pdf"
OUTPUT_PNG = FIG_DIR / "vqe_active_orbital_gallery.png"
ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
GRID_POINTS = 34
ISO_FRACTION = 0.22
BASE_PADDING = 2.6
BASE_MINIMUM_SPAN = 5.8
MAX_GRID_EXPANSIONS = 4
GRID_EXPANSION_FACTOR = 1.35

PHASE_POSITIVE = "#76B900"
PHASE_NEGATIVE = "#b14f4f"
PANEL_EDGE = "#c7d0db"
PANEL_ROUNDING = 0.035

ATOM_COLORS = {
    "H": "#f5f3ec",
    "B": "#ba8c63",
    "Li": "#b182ff",
    "Be": "#3f88c5",
    "O": "#d95757",
    "N": "#3a86c8",
}

ATOM_RADII = {
    "H": 0.25,
    "B": 0.36,
    "Li": 0.48,
    "Be": 0.38,
    "O": 0.30,
    "N": 0.31,
}

COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "Li": 1.28,
    "Be": 0.96,
    "O": 0.66,
    "N": 0.71,
}


@dataclass(frozen=True)
class MoleculeSpec:
    key: str
    label_tex: str
    geometry: tuple[tuple[str, tuple[float, float, float]], ...]
    active_electrons: int
    active_orbitals: int
    basis: str = "sto-3g"
    charge: int = 0
    multiplicity: int = 1

    @property
    def n_qubits(self) -> int:
        return 2 * self.active_orbitals


VQE_SPECS = (
    MoleculeSpec(
        key="bh",
        label_tex=r"$\mathrm{BH}$",
        geometry=(("B", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.23))),
        active_electrons=2,
        active_orbitals=3,
    ),
    MoleculeSpec(
        key="lih",
        label_tex=r"$\mathrm{LiH}$",
        geometry=(("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.57))),
        active_electrons=2,
        active_orbitals=4,
    ),
    MoleculeSpec(
        key="beh2",
        label_tex=r"$\mathrm{BeH_2}$",
        geometry=(("H", (0.0, 0.0, -1.326)), ("Be", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.326))),
        active_electrons=4,
        active_orbitals=4,
    ),
    MoleculeSpec(
        key="h2o",
        label_tex=r"$\mathrm{H_2O}$",
        geometry=(("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.757, 0.587)), ("H", (0.0, -0.757, 0.587))),
        active_electrons=6,
        active_orbitals=4,
    ),
)


def configure_style() -> None:
    apply_style()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
            "font.family": "serif",
            "font.size": 20,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def build_molecule(spec: MoleculeSpec):
    atom_spec = "; ".join(
        f"{symbol} {coords[0]} {coords[1]} {coords[2]}" for symbol, coords in spec.geometry
    )
    mol = gto.M(
        atom=atom_spec,
        basis=spec.basis,
        unit="Angstrom",
        charge=spec.charge,
        spin=spec.multiplicity - 1,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 256
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for {spec.key}")
    return mol, mf


def active_indices(mol, spec: MoleculeSpec) -> list[int]:
    core = (mol.nelectron - spec.active_electrons) // 2
    return list(range(core, core + spec.active_orbitals))


def orbital_grid(atom_coords: np.ndarray, padding: float, minimum_span: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = atom_coords.min(axis=0) - padding
    maxs = atom_coords.max(axis=0) + padding
    spans = maxs - mins
    for axis in range(3):
        if spans[axis] < minimum_span:
            extra = 0.5 * (minimum_span - spans[axis])
            mins[axis] -= extra
            maxs[axis] += extra
    xs = np.linspace(mins[0], maxs[0], GRID_POINTS)
    ys = np.linspace(mins[1], maxs[1], GRID_POINTS)
    zs = np.linspace(mins[2], maxs[2], GRID_POINTS)
    return xs, ys, zs


def extract_surface(values: np.ndarray, level: float, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, _normals, _ = measure.marching_cubes(
        values,
        level=level,
        spacing=(xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]),
    )
    verts[:, 0] += xs[0]
    verts[:, 1] += ys[0]
    verts[:, 2] += zs[0]
    return verts, faces


def surface_touches_boundary(verts: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> bool:
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    upper = np.asarray([xs[-1], ys[-1], zs[-1]], dtype=float)
    lower = np.asarray([xs[0], ys[0], zs[0]], dtype=float)
    spacing = np.asarray([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]], dtype=float)
    tolerance = 0.6 * spacing
    return bool(np.any(mins <= lower + tolerance) or np.any(maxs >= upper - tolerance))


def orbital_surfaces(mol, mf, indices: list[int]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    atom_coords = mol.atom_coords()
    center = atom_coords.mean(axis=0)
    padding = BASE_PADDING
    minimum_span = BASE_MINIMUM_SPAN
    chosen_surfaces: list[dict] = []
    bounds = atom_coords
    for attempt in range(MAX_GRID_EXPANSIONS):
        xs, ys, zs = orbital_grid(atom_coords, padding, minimum_span)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        grid_coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        ao_values = mol.eval_gto("GTOval_sph", grid_coords)
        surfaces: list[dict] = []
        all_points: list[np.ndarray] = [atom_coords - center]
        clipped = False
        for local_idx, mo_idx in enumerate(indices):
            values = (ao_values @ mf.mo_coeff[:, mo_idx]).reshape(xx.shape)
            max_abs = float(np.max(np.abs(values)))
            if not np.isfinite(max_abs) or max_abs <= 1e-8:
                surfaces.append({"positive": None, "negative": None, "label": local_idx + 1})
                continue
            level = ISO_FRACTION * max_abs
            positive = None
            negative = None
            if float(values.max()) > level:
                verts, faces = extract_surface(values, level, xs, ys, zs)
                clipped = clipped or surface_touches_boundary(verts, xs, ys, zs)
                positive = (verts - center, faces)
                all_points.append(verts - center)
            if float((-values).max()) > level:
                verts, faces = extract_surface(-values, level, xs, ys, zs)
                clipped = clipped or surface_touches_boundary(verts, xs, ys, zs)
                negative = (verts - center, faces)
                all_points.append(verts - center)
            surfaces.append({"positive": positive, "negative": negative, "label": local_idx + 1})
        chosen_surfaces = surfaces
        bounds = np.vstack(all_points)
        if not clipped:
            break
        padding *= GRID_EXPANSION_FACTOR
        minimum_span *= GRID_EXPANSION_FACTOR
    atom_coords = atom_coords - center
    return chosen_surfaces, atom_coords, bounds


def draw_sphere(ax, center: np.ndarray, radius: float, color: str) -> None:
    u_vals = np.linspace(0.0, 2.0 * math.pi, 22)
    v_vals = np.linspace(0.0, math.pi, 16)
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
        alpha=1.0,
    )
    surface.set_rasterized(True)


def draw_bonds(ax, symbols: list[str], atom_coords: np.ndarray) -> None:
    for idx in range(len(symbols)):
        for jdx in range(idx + 1, len(symbols)):
            threshold = 1.26 * (COVALENT_RADII[symbols[idx]] + COVALENT_RADII[symbols[jdx]]) * ANGSTROM_TO_BOHR
            distance = float(np.linalg.norm(atom_coords[idx] - atom_coords[jdx]))
            if distance > threshold:
                continue
            ax.plot(
                [atom_coords[idx, 0], atom_coords[jdx, 0]],
                [atom_coords[idx, 1], atom_coords[jdx, 1]],
                [atom_coords[idx, 2], atom_coords[jdx, 2]],
                color="#57606a",
                linewidth=2.1,
                alpha=0.9,
                solid_capstyle="round",
            )


def add_surface(ax, payload: tuple[np.ndarray, np.ndarray] | None, color: str) -> None:
    if payload is None:
        return
    verts, faces = payload
    collection = Poly3DCollection(
        verts[faces],
        facecolor=color,
        edgecolor="none",
        linewidths=0.0,
        alpha=0.54,
    )
    collection.set_rasterized(True)
    ax.add_collection3d(collection)


def style_orbital_axis(ax, bounds: np.ndarray) -> None:
    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
    ax.grid(False)
    mins = bounds.min(axis=0)
    maxs = bounds.max(axis=0)
    span = float(max((maxs - mins).max(), 1.0))
    center = 0.5 * (mins + maxs)
    half = 0.47 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")
    ax.view_init(elev=18, azim=34)


def title_for(spec: MoleculeSpec) -> str:
    return (
        rf"{spec.label_tex} $\cdot$ "
        rf"$\mathrm{{CAS}}({spec.active_electrons},{spec.active_orbitals})$ $\cdot$ "
        rf"${spec.n_qubits}\ \mathrm{{qubits}}$"
    )


def render_molecule_panel(subfig, spec: MoleculeSpec) -> None:
    mol, mf = build_molecule(spec)
    symbols = [symbol for symbol, _coords in spec.geometry]
    active = active_indices(mol, spec)
    surfaces, atom_coords, bounds = orbital_surfaces(mol, mf, active)
    subfig.patch.set_alpha(0.0)
    panel = FancyBboxPatch(
        (0.018, 0.028),
        0.964,
        0.944,
        boxstyle=rf"round,pad=0.008,rounding_size={PANEL_ROUNDING}",
        transform=subfig.transSubfigure,
        facecolor="#ffffff",
        edgecolor=PANEL_EDGE,
        linewidth=1.55,
        clip_on=False,
        zorder=-10,
    )
    subfig.add_artist(panel)
    axes = subfig.subplots(
        1,
        spec.active_orbitals,
        subplot_kw={"projection": "3d"},
        gridspec_kw={"wspace": 0.02},
    )
    subfig.subplots_adjust(left=0.055, right=0.945, top=0.845, bottom=0.115, wspace=0.01)
    axes_list = list(np.atleast_1d(axes))
    subfig.suptitle(title_for(spec), y=0.955, fontsize=22)
    for ax, surface_data in zip(axes_list, surfaces):
        add_surface(ax, surface_data["negative"], PHASE_NEGATIVE)
        add_surface(ax, surface_data["positive"], PHASE_POSITIVE)
        draw_bonds(ax, symbols, atom_coords)
        for symbol, center in zip(symbols, atom_coords):
            radius = ATOM_RADII.get(symbol, 0.3) * ANGSTROM_TO_BOHR
            draw_sphere(ax, center, radius, ATOM_COLORS.get(symbol, "#cccccc"))
        style_orbital_axis(ax, bounds)
        ax.text2D(
            0.5,
            0.88,
            rf"$\phi_{{{surface_data['label']}}}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=16,
        )


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    fig = plt.figure(figsize=(22.8, 8.9))
    grid = fig.subfigures(2, 2, wspace=0.03, hspace=0.04)
    ordered_subfigs = [grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]]
    for subfig, spec in zip(ordered_subfigs, VQE_SPECS):
        render_molecule_panel(subfig, spec)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUTPUT_PNG, dpi=260, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(json.dumps({"pdf": str(OUTPUT_PDF), "png": str(OUTPUT_PNG)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
