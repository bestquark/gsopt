# Proposed Periodic AFQMC Targets

The current validated AFQMC lane in `examples/afqmc/` is still the small-molecule
set used by the paper figures. The next nonmolecular extension should move to a
compact periodic-electronic set that spans clearly different regimes under the
same fixed-budget `PySCF + ipie` evaluator.

The four proposed first periodic targets are:

| Slug | Target | Why it belongs in the next benchmark family |
| --- | --- | --- |
| `heg_14e_rs1_gamma` | 3D homogeneous electron gas with 14 electrons at fixed `r_s` and Gamma twist | Metallic, basis-clean, and free of electron-ion pseudopotentials. This is the cleanest periodic AFQMC target for testing trial-state and propagation-policy search. |
| `h10_chain_pbc` | Periodic H10 hydrogen chain supercell | Adds periodicity to a bond-stretching, near-degenerate system where trial quality matters strongly but the chemistry remains simple. |
| `lih_rocksalt_prim` | LiH rocksalt primitive cell | A compact ionic insulator that introduces realistic electron-ion structure while staying small enough for repeated fixed-budget scoring. |
| `diamond_prim` | Diamond primitive cell | A canonical covalent semiconductor benchmark that complements LiH and the HEG with a different nodal and basis-sensitivity profile. |

This set was chosen to span:

- a metallic electron gas
- a quasi-one-dimensional correlated hydrogen system
- an ionic insulator
- a covalent semiconductor

That diversity is important because AFQMC search knobs such as trial family,
orbital basis, Cholesky threshold, timestep, and walker budget can behave very
differently across those regimes.
