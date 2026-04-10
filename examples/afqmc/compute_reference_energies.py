from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pyscf import fci, gto, scf

from model_registry import ACTIVE_MOLECULES, MOLECULE_SPECS

OUTPUT_PATH = Path(__file__).resolve().parent / "reference_energies.json"


def compute_reference(molecule: str) -> dict:
    spec = MOLECULE_SPECS[molecule]
    start = time.perf_counter()
    mol = gto.M(
        atom=spec.atom,
        basis=spec.basis,
        charge=spec.charge,
        spin=spec.spin,
        unit=spec.unit,
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    cisolver = fci.FCI(mf)
    energy, _ = cisolver.kernel()
    wall = time.perf_counter() - start
    return {
        "reference_energy": float(energy),
        "reference_method": "FCI",
        "basis": spec.basis,
        "charge": spec.charge,
        "spin": spec.spin,
        "wall_seconds": wall,
        "hf_energy": float(mf.e_tot),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute frozen AFQMC molecular reference energies.")
    parser.add_argument("--molecule", choices=ACTIVE_MOLECULES)
    args = parser.parse_args()

    molecules = [args.molecule] if args.molecule else list(ACTIVE_MOLECULES)
    existing = {}
    if OUTPUT_PATH.exists() and OUTPUT_PATH.read_text().strip():
        existing = json.loads(OUTPUT_PATH.read_text())
    for molecule in molecules:
        print(f"computing reference for {molecule}...", flush=True)
        existing[molecule] = compute_reference(molecule)
    OUTPUT_PATH.write_text(json.dumps(existing, indent=2))
    print(json.dumps({"reference_file": str(OUTPUT_PATH), "molecules": molecules}, indent=2))


if __name__ == "__main__":
    main()
