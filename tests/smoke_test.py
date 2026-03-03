#!/usr/bin/env python3
"""Quick smoke test for PRISM core functionality."""
import numpy as np

print("=== PRISM Smoke Test ===\n")

# 1. Symmetry groups
from prism import SymmetryGroup
for name in ["T", "O", "I"]:
    sg = SymmetryGroup.from_name(name)
    axes = sg.get_symmetry_axes()
    print(f"{name}: {sg.order} ops, axes: {list(axes.keys())}")

# 2. Expansion
sg = SymmetryGroup.from_name("T")
pt = np.array([[10.0, 0.0, 0.0]])
exp = sg.expand_coords(pt)
assert exp.shape == (12, 3)
assert np.allclose(np.linalg.norm(exp, axis=1), 10.0)
print(f"\nExpansion: 1 point -> {len(exp)} copies OK")

# 3. Cage spec
from prism.core.cage import CageSpec, CavitySpec, CageDesign
spec = CageSpec(symmetry_group="T", cavity=CavitySpec(target_diameter_nm=10.0, crystal_phase="Fe3O4_magnetite"))
d = CageDesign(spec=spec)
print(f"\n{d.summary()}")

# 4. Surface chemistry
from prism.core.residue_surface import SurfaceChemSpec
sc = SurfaceChemSpec.for_phase("Fe3O4_magnetite")
print(f'\nPhase: {sc.target_phase}, preferred: {sc.preferred_residues}')

# 5. Lattice
from prism.assembly.lattice import LatticeBuilder, LatticeSpec
ls = LatticeSpec(lattice_type="SC", lattice_constant=50.0, repeats=(2, 2, 2))
lb = LatticeBuilder(ls)
centers = lb.get_cage_centers()
assert len(centers) == 8
print(f"\nSC 2x2x2: {len(centers)} positions OK")

# 6. Cavity analysis
from prism.analysis.cavity_analysis import analyse_cavity
n = 300
phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
theta = np.pi * (1 + 5**0.5) * np.arange(n)
shell = np.column_stack([30 * np.sin(phi) * np.cos(theta), 30 * np.sin(phi) * np.sin(theta), 30 * np.cos(phi)])
report = analyse_cavity(shell, n_samples=50000)
print(f"\nCavity: {report.volume_nm3:.1f} nm3, void: {report.void_fraction:.2%}")

# 7. Clash check
from prism.analysis.clash_check import check_clashes
a = np.array([[0.0, 0.0, 0.0]])
b = np.array([[100.0, 0.0, 0.0]])
r = check_clashes(a, b)
assert r.passed
print(f"\nClash check (distant): PASSED")

# 8. Metrics
from prism.analysis.metrics import symmetry_rmsd
coords = np.random.randn(50, 3)
rmsd = symmetry_rmsd([coords, coords])
assert rmsd < 1e-6
print(f"Symmetry RMSD (identical): {rmsd:.6f}")

# 9. Project I/O
import tempfile
from prism.io.project import PRISMProject
with tempfile.TemporaryDirectory() as td:
    proj = PRISMProject.create(f"{td}/test", name="Test")
    proj.set_spec(spec)
    proj.save()
    loaded = PRISMProject.load(f"{td}/test")
    assert loaded.spec["symmetry_group"] == "T"
    print(f"\nProject I/O: round-trip OK")

# 10. PDB I/O
from prism.io.pdb import write_pdb, write_fasta, read_fasta
with tempfile.TemporaryDirectory() as td:
    write_pdb(np.array([[1.0, 2.0, 3.0]]), f"{td}/test.pdb")
    write_fasta({"seq1": "ACDEF"}, f"{td}/test.fasta")
    seqs = read_fasta(f"{td}/test.fasta")
    assert seqs["seq1"] == "ACDEF"
    print("PDB/FASTA I/O: OK")

print("\n=== ALL SMOKE TESTS PASSED ===")
