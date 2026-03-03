# PRISM

**Programmable protein cages as a universal nanocrystal manufacturing platform**

PRISM is a computational design toolkit for engineering de novo protein cages that
template deterministic nanocrystal formation. It integrates with RFdiffusion for
backbone generation, BindCraft for interface design, and ProteinMPNN for sequence
optimisation — all tied together with a high-performance Rust geometry core and
interactive 3D visualisation.

## Three Tunable Parameters

1. **Interior Geometry** — Controls nanocrystal size (5–20 nm cavities)
2. **Surface Chemistry** — Controls which crystal phase nucleates
3. **Exterior Interfaces** — Controls lattice architecture for spatial organisation

## Installation

```bash
# Requires Rust toolchain and Python ≥ 3.10
pip install maturin
maturin develop --release
```

## Quick Start

```python
from prism.core.cage import CageDesign, CavitySpec
from prism.core.symmetry import SymmetryGroup
from prism.viz.viewer import CageViewer

# Define a cage specification
cage = CageDesign(
    symmetry_group="I",
    target_cavity=CavitySpec(target_diameter_nm=10.0, shape="spherical"),
)

# Expand symmetry and visualise
sym = SymmetryGroup.from_name("I")
viewer = CageViewer(cage)
viewer.show_full_cage()
```

## Project Structure

```
prism/
├── rust/              # Rust geometry core (PyO3 → prism._rust_core)
│   └── src/
│       ├── geometry/  # Symmetry, polyhedra, cavity, transforms
│       ├── spatial/   # KD-tree, SASA
│       └── lattice/   # Docking, assembly
├── python/prism/      # Python package
│   ├── core/          # CageDesign, Structure, SymmetryGroup
│   ├── design/        # RFdiffusion, BindCraft, ProteinMPNN wrappers
│   ├── assembly/      # Docking interface design, lattice builder
│   ├── analysis/      # Cavity analysis, clash detection, metrics
│   ├── viz/           # py3Dmol viewer, symmetry visualisation
│   ├── io/            # PDB/mmCIF I/O, project serialisation
│   └── cli/           # Typer CLI
├── notebooks/         # Demo Jupyter notebooks
├── tests/             # pytest test suite
└── data/              # Symmetry matrices, example PDBs
```

## Design Pipeline

```
Specify → Generate (RFdiffusion) → Sequence (ProteinMPNN) → Validate (AF2)
    → Analyse cavity → Design interfaces (BindCraft) → Assemble lattice → Visualise
```

## License

MIT
