#!/usr/bin/env python3
"""
PRISM Demo — Cage Geometry & Design Workflow
=============================================

This notebook-style script demonstrates the core PRISM toolkit:

1. Symmetry group exploration (T, O, I)
2. Cage specification & design setup
3. Cavity analysis on synthetic geometry
4. Surface chemistry assignment
5. Lattice construction
6. Quality metrics
7. 3D visualization (in Jupyter)

Run interactively:  python -i notebooks/01_cage_geometry_demo.py
Or as Jupyter:      jupyter notebook  (convert to .ipynb)
"""

# %% [markdown]
# # PRISM — Cage Geometry Demo

# %% Imports
import numpy as np
from prism import SymmetryGroup, CageSpec, CavitySpec, CageDesign
from prism.core.residue_surface import SurfaceChemSpec, PHASE_RESIDUE_MAP

print("PRISM imported successfully!")
print(f"Available crystal phases: {list(PHASE_RESIDUE_MAP.keys())}")

# %% [markdown]
# ## 1. Symmetry Groups
# PRISM supports three polyhedral symmetry groups for cage design:
# - **Tetrahedral (T)**: 12 symmetry operations, 4 C3 + 3 C2 axes
# - **Octahedral (O)**: 24 operations, 3 C4 + 4 C3 + 6 C2 axes
# - **Icosahedral (I)**: 60 operations, 6 C5 + 10 C3 + 15 C2 axes

# %% Explore symmetry groups
for name in ["T", "O", "I"]:
    sg = SymmetryGroup.from_name(name)
    axes = sg.get_symmetry_axes()
    print(f"\n{name} group: {sg.order} operations")
    for order, axis_list in axes.items():
        print(f"  C{order} axes: {len(axis_list)}")

# %% Symmetry expansion demo
print("\n--- Symmetry Expansion ---")
sg_T = SymmetryGroup.from_name("T")

# Place a single point at (10, 0, 0)
point = np.array([[10.0, 0.0, 0.0]])
expanded = sg_T.expand_coords(point)

print(f"Input:  1 point → Output: {expanded.shape[0]} symmetry copies")
print(f"All at distance 10.0 from origin: {np.allclose(np.linalg.norm(expanded, axis=1), 10.0)}")

# %% Multi-atom expansion with chain IDs
subunit = np.array([
    [10.0, 0.0, 0.0],
    [10.5, 0.5, 0.0],
    [11.0, 0.0, 0.5],
    [10.0, -0.5, 0.5],
])

expanded, chain_ids = sg_T.expand_with_chain_ids(subunit)
print(f"\nSubunit: {len(subunit)} atoms × {sg_T.order} copies = {len(expanded)} total atoms")
print(f"Unique chains: {len(set(chain_ids))}")

# %% [markdown]
# ## 2. Cage Specification
# A PRISM cage design is defined by three parameters:
# - **Interior geometry**: cavity size (5-20 nm diameter)
# - **Surface chemistry**: crystal phase (determines residue composition)
# - **Exterior interfaces**: lattice architecture for 3D assembly

# %% Create a cage specification
spec = CageSpec(
    symmetry_group="T",
    cavity=CavitySpec(
        target_diameter_nm=10.0,
        crystal_phase="Fe3O4_magnetite",
    ),
    subunit_length_range=(80, 120),
)

print("Cage Specification:")
print(f"  Symmetry: {spec.symmetry_group}")
print(f"  Target diameter: {spec.cavity.target_diameter_nm} nm")
print(f"  Crystal phase: {spec.cavity.crystal_phase}")
print(f"  Subunit length: {spec.subunit_length_range} residues")

# %% Create a CageDesign
design = CageDesign(spec=spec)
print(f"\n{design.summary()}")

# %% [markdown]
# ## 3. Surface Chemistry
# Each crystal phase requires specific residue types on the interior
# surface to nucleate the target material.

# %% Surface chemistry specifications
print("\n--- Crystal Phase → Preferred Interior Residues ---")
for phase, info in PHASE_RESIDUE_MAP.items():
    spec = SurfaceChemSpec.for_phase(phase)
    print(f"\n{phase}:")
    print(f"  Coordinating: {info['coordinating_residues']}")
    print(f"  Geometry:     {info['coordination_geometry']}")
    print(f"  Preferred:    {spec.preferred_residues}")

# %% [markdown]
# ## 4. Cavity Analysis (Synthetic Demo)
# Demonstrate cavity analysis on a synthetic hollow sphere.

# %% Create synthetic cage shell
from prism.analysis.cavity_analysis import analyse_cavity

np.random.seed(42)
n_atoms = 500
r_shell = 50.0  # 50 Å = 5 nm radius (10 nm diameter)

# Golden spiral distribution on sphere
indices = np.arange(n_atoms) + 0.5
phi = np.arccos(1 - 2 * indices / n_atoms)
theta = np.pi * (1 + 5**0.5) * indices

shell_coords = np.column_stack([
    r_shell * np.sin(phi) * np.cos(theta),
    r_shell * np.sin(phi) * np.sin(theta),
    r_shell * np.cos(phi),
])

print(f"Synthetic cage: {n_atoms} atoms on a {2 * r_shell / 10:.0f} nm diameter shell")

# %% Run cavity analysis
report = analyse_cavity(
    shell_coords,
    target_diameter_nm=10.0,
    n_samples=100_000,
)

print(report.summary())

# %% [markdown]
# ## 5. Lattice Construction
# PRISM cages self-assemble into 3D lattices through designed
# docking interfaces on their exterior surfaces.

# %% Build lattices
from prism.assembly.lattice import LatticeBuilder, LatticeSpec

print("\n--- Lattice Architectures ---")
for lattice_type in ["SC", "BCC", "FCC"]:
    spec = LatticeSpec(
        lattice_type=lattice_type,
        lattice_constant=100.0,  # 10 nm spacing in Å
        repeats=(3, 3, 3),
    )
    builder = LatticeBuilder(spec)
    centers = builder.get_cage_centers()
    size = builder.estimate_lattice_size()

    print(f"\n{lattice_type} lattice (3×3×3):")
    print(f"  Cage positions: {len(centers)}")
    dims = size["dimensions_nm"]
    print(f"  Lattice extent: {dims[0]:.0f} × {dims[1]:.0f} × {dims[2]:.0f} nm")

# %% [markdown]
# ## 6. Quality Metrics

# %% Compute metrics on synthetic data
from prism.analysis.metrics import symmetry_rmsd, packing_density

# Simulate symmetry-related subunits with small noise
base_subunit = np.random.randn(50, 3) * 5 + np.array([20, 0, 0])
sg = SymmetryGroup.from_name("T")

subunits = []
for op in sg.operations:
    rotated = (op @ base_subunit.T).T
    noise = np.random.randn(*rotated.shape) * 0.1  # 0.1 Å noise
    subunits.append(rotated + noise)

rmsd = symmetry_rmsd(subunits)
print(f"\nSymmetry RMSD (with 0.1 Å noise): {rmsd:.3f} Å")

# Packing density of the full assembly
all_coords = np.vstack(subunits)
pd = packing_density(all_coords)
print(f"Packing density: {pd:.3f}")

# %% [markdown]
# ## 7. Clash Detection

# %% Test clash detection
from prism.analysis.clash_check import check_clashes

# Two non-overlapping groups
coords_a = np.random.randn(100, 3) * 5
coords_b = np.random.randn(100, 3) * 5 + np.array([50, 0, 0])  # 50 Å apart

report = check_clashes(coords_a, coords_b, cutoff=2.0)
print(f"\nDistant structures: {report.summary()}")

# Overlapping groups
coords_c = np.random.randn(100, 3) * 5
coords_d = np.random.randn(100, 3) * 5  # Same region

report2 = check_clashes(coords_c, coords_d, cutoff=2.0)
print(f"\nOverlapping structures: {report2.summary()}")

# %% [markdown]
# ## 8. Project Management

# %% Create a PRISM project
import tempfile
from prism.io.project import PRISMProject
from prism.core.cage import CageSpec, CavitySpec

with tempfile.TemporaryDirectory() as tmpdir:
    proj = PRISMProject.create(f"{tmpdir}/demo_Fe3O4_cage", name="Fe₃O₄ Magnetite Cage")

    proj.set_spec(CageSpec(
        symmetry_group="T",
        cavity=CavitySpec(target_diameter_nm=10.0, crystal_phase="Fe3O4_magnetite"),
    ))
    proj.set_parameter("design_method", "RFdiffusion")
    proj.add_note("Demo project for Fe₃O₄ nanocrystal templating")
    proj.snapshot("Initial setup", metrics={"target_nm": 10.0})

    print(proj.summary())

# %% [markdown]
# ## 9. Design Pipeline Overview
#
# The full PRISM workflow:
# ```
# 1. Define CageSpec (symmetry, cavity size, crystal phase)
# 2. Design subunit backbone → RFdiffusion (symmetry mode)
# 3. Design sequences → ProteinMPNN
# 4. Design surface chemistry → interior residue mutation
# 5. Design docking interfaces → BindCraft
# 6. Build lattice → assembly module
# 7. Analyse → cavity, clashes, metrics
# 8. Iterate → refine based on metrics
# ```
#
# CLI equivalent:
# ```bash
# prism init my_cage --symmetry T --diameter 10 --phase Fe3O4_magnetite
# prism design subunit --project my_cage --rfdiffusion-dir /path/to/RFdiffusion
# prism sequence --project my_cage --mpnn-dir /path/to/ProteinMPNN
# prism analyze --project my_cage
# ```

# %% [markdown]
# ## 10. Visualization (Jupyter only)
#
# In a Jupyter notebook, use the built-in viewer:
# ```python
# from prism.viz.viewer import CageViewer, quick_view
#
# # Quick view of a structure
# quick_view("my_cage.pdb")
#
# # Detailed view with symmetry axes
# v = CageViewer()
# v.show_full_cage(pdb_path="my_cage.pdb")
# v.show_symmetry_axes("T")
# v.show()
# ```

print("\n" + "=" * 60)
print("PRISM demo complete!")
print("=" * 60)
