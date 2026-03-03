"""Tests for symmetry group operations."""

import numpy as np
import pytest


class TestSymmetryGroupPython:
    """Test the pure-Python fallback symmetry implementation."""

    def test_tetrahedral_group_size(self):
        from prism.core.symmetry import SymmetryGroup
        sg = SymmetryGroup.from_name("T")
        assert sg.order == 12

    def test_octahedral_group_size(self):
        from prism.core.symmetry import SymmetryGroup
        sg = SymmetryGroup.from_name("O")
        assert sg.order == 24

    def test_icosahedral_group_size(self):
        from prism.core.symmetry import SymmetryGroup
        sg = SymmetryGroup.from_name("I")
        assert sg.order == 60

    def test_identity_in_group(self):
        from prism.core.symmetry import SymmetryGroup
        sg = SymmetryGroup.from_name("T")
        identity = np.eye(3)
        found = any(
            np.allclose(op, identity, atol=1e-10)
            for op in sg.operations
        )
        assert found, "Identity matrix not found in tetrahedral group"

    def test_operations_are_orthogonal(self):
        from prism.core.symmetry import SymmetryGroup
        for name in ["T", "O", "I"]:
            sg = SymmetryGroup.from_name(name)
            for op in sg.operations:
                # R @ R.T should be identity
                product = op @ op.T
                assert np.allclose(product, np.eye(3), atol=1e-10), \
                    f"Non-orthogonal operation in {name} group"

    def test_operations_have_det_1(self):
        from prism.core.symmetry import SymmetryGroup
        for name in ["T", "O", "I"]:
            sg = SymmetryGroup.from_name(name)
            for op in sg.operations:
                det = np.linalg.det(op)
                assert abs(det - 1.0) < 1e-10, \
                    f"Operation in {name} group has det={det}"

    def test_group_closure(self):
        from prism.core.symmetry import SymmetryGroup
        sg = SymmetryGroup.from_name("T")

        # Product of any two elements should be in the group
        for i in range(min(sg.order, 5)):
            for j in range(min(sg.order, 5)):
                product = sg.operations[i] @ sg.operations[j]
                found = any(
                    np.allclose(product, op, atol=1e-10)
                    for op in sg.operations
                )
                assert found, "Closure violated in tetrahedral group"

    def test_expand_coords(self):
        from prism.core.symmetry import SymmetryGroup
        sg = SymmetryGroup.from_name("T")

        # Single point
        point = np.array([[1.0, 0.0, 0.0]])
        expanded = sg.expand_coords(point)

        # Should get 12 copies
        assert expanded.shape == (12, 3)

        # All should be unit distance from origin
        norms = np.linalg.norm(expanded, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_expand_with_chain_ids(self):
        from prism.core.symmetry import SymmetryGroup
        sg = SymmetryGroup.from_name("O")

        coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        expanded, chain_ids = sg.expand_with_chain_ids(coords)

        assert expanded.shape == (48, 3)  # 24 ops × 2 points
        assert len(chain_ids) == 48
        assert len(set(chain_ids)) == 24  # 24 unique chains

    def test_symmetry_axes(self):
        from prism.core.symmetry import SymmetryGroup
        sg = SymmetryGroup.from_name("T")
        axes = sg.get_symmetry_axes()

        assert "C3" in axes  # C3 axes
        assert "C2" in axes  # C2 axes

    def test_invalid_group_raises(self):
        from prism.core.symmetry import SymmetryGroup
        with pytest.raises(ValueError):
            SymmetryGroup.from_name("X")


class TestCavityAnalysis:
    """Test cavity analysis module."""

    def test_cavity_of_hollow_sphere(self):
        """A hollow arrangement of atoms should have non-zero cavity."""
        from prism.analysis.cavity_analysis import analyse_cavity

        # Generate a sphere of atoms
        n_atoms = 200
        phi = np.random.uniform(0, 2 * np.pi, n_atoms)
        theta = np.random.uniform(0, np.pi, n_atoms)
        r = 20.0  # 20 Å radius sphere

        coords = np.column_stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ])

        report = analyse_cavity(coords, n_samples=50_000)

        assert report.volume_angstrom3 > 0
        assert report.void_fraction > 0
        assert report.inscribed_radius > 0

    def test_cavity_report_summary(self):
        from prism.analysis.cavity_analysis import CavityReport
        report = CavityReport(
            volume_angstrom3=10000.0,
            volume_nm3=10.0,
            inscribed_radius=15.0,
            effective_diameter_nm=3.0,
            void_fraction=0.5,
        )
        summary = report.summary()
        assert "10,000" in summary or "10000" in summary
        assert "nm" in summary


class TestClashCheck:
    """Test clash detection."""

    def test_no_clashes_distant(self):
        from prism.analysis.clash_check import check_clashes

        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = np.array([[100.0, 100.0, 100.0]])

        report = check_clashes(a, b, cutoff=2.0)
        assert report.passed
        assert report.n_clashes == 0

    def test_clashes_overlapping(self):
        from prism.analysis.clash_check import check_clashes

        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[0.5, 0.0, 0.0]])

        report = check_clashes(a, b, cutoff=2.0)
        assert not report.passed
        assert report.n_clashes > 0

    def test_find_contacts(self):
        from prism.analysis.clash_check import find_contacts

        a = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        b = np.array([[3.0, 0.0, 0.0], [20.0, 0.0, 0.0]])

        contacts = find_contacts(a, b, cutoff=4.0)
        assert len(contacts) >= 1  # 0→0 should be within 4 Å (dist=3)


class TestStructure:
    """Test ProteinStructure wrapper (no actual PDB files needed)."""

    def test_create_empty(self):
        from prism.core.structure import ProteinStructure
        ps = ProteinStructure.__new__(ProteinStructure)
        ps._structure = None
        ps._path = None
        assert ps._structure is None


class TestResidueSurface:
    """Test surface chemistry specification."""

    def test_phase_map_exists(self):
        from prism.core.residue_surface import PHASE_RESIDUE_MAP
        assert "Fe3O4_magnetite" in PHASE_RESIDUE_MAP
        assert "Fe16N2" in PHASE_RESIDUE_MAP

    def test_spec_from_phase(self):
        from prism.core.residue_surface import SurfaceChemSpec
        spec = SurfaceChemSpec.for_phase("Fe3O4_magnetite")
        assert spec.target_phase == "Fe3O4_magnetite"
        assert len(spec.preferred_residues) > 0

    def test_spec_score(self):
        from prism.core.residue_surface import SurfaceChemSpec
        spec = SurfaceChemSpec.for_phase("Fe3O4_magnetite")
        # Score returns a dict with fraction_correct
        result = spec.score()
        assert result["fraction_correct"] == 0.0

    def test_unknown_phase_raises(self):
        from prism.core.residue_surface import SurfaceChemSpec
        with pytest.raises(ValueError):
            SurfaceChemSpec.for_phase("unobtainium")


class TestCageSpec:
    """Test Pydantic cage spec models."""

    def test_cage_spec_serialisation(self):
        from prism.core.cage import CageSpec, CavitySpec

        spec = CageSpec(
            symmetry_group="T",
            cavity=CavitySpec(
                target_diameter_nm=10.0,
                crystal_phase="Fe3O4_magnetite",
            ),
            subunit_length_range=(80, 120),
        )

        data = spec.model_dump()
        assert data["symmetry_group"] == "T"
        assert data["cavity"]["target_diameter_nm"] == 10.0

        # Round-trip
        spec2 = CageSpec(**data)
        assert spec2.symmetry_group == spec.symmetry_group

    def test_cage_design_summary(self):
        from prism.core.cage import CageDesign, CageSpec, CavitySpec

        spec = CageSpec(
            symmetry_group="O",
            cavity=CavitySpec(target_diameter_nm=8.0),
        )
        design = CageDesign(spec=spec)
        summary = design.summary()
        assert "Octahedral" in summary or "O" in summary


class TestLattice:
    """Test lattice builder."""

    def test_sc_lattice(self):
        from prism.assembly.lattice import LatticeBuilder, LatticeSpec

        spec = LatticeSpec(lattice_type="SC", lattice_constant=50.0, repeats=(2, 2, 2))
        builder = LatticeBuilder(spec)
        centers = builder.get_cage_centers()

        assert len(centers) == 8  # 2×2×2

    def test_bcc_lattice(self):
        from prism.assembly.lattice import LatticeBuilder, LatticeSpec

        spec = LatticeSpec(lattice_type="BCC", lattice_constant=50.0, repeats=(2, 2, 2))
        builder = LatticeBuilder(spec)
        centers = builder.get_cage_centers()

        # BCC primitive cell gives 2x2x2 = 8 lattice points
        assert len(centers) == 8

    def test_lattice_size_estimate(self):
        from prism.assembly.lattice import LatticeBuilder, LatticeSpec

        spec = LatticeSpec(lattice_type="FCC", lattice_constant=40.0, repeats=(3, 3, 3))
        builder = LatticeBuilder(spec)
        size = builder.estimate_lattice_size()

        assert size["n_cells"] == 27
        assert all(d > 0 for d in size["dimensions_angstrom"])


class TestMetrics:
    """Test quality metrics."""

    def test_symmetry_rmsd_identical(self):
        from prism.analysis.metrics import symmetry_rmsd

        coords = np.random.randn(50, 3)
        rmsd = symmetry_rmsd([coords, coords, coords])
        assert rmsd < 1e-6

    def test_packing_density_range(self):
        from prism.analysis.metrics import packing_density

        # Random atoms in a sphere
        coords = np.random.randn(500, 3) * 10
        pd = packing_density(coords)
        assert 0.0 <= pd <= 1.0


class TestProject:
    """Test project I/O."""

    def test_create_and_load(self, tmp_path):
        from prism.io.project import PRISMProject

        proj_dir = tmp_path / "test_project"
        proj = PRISMProject.create(str(proj_dir), name="Test")

        assert (proj_dir / "prism_project.json").exists()
        assert proj.structures_dir.exists()

        loaded = PRISMProject.load(str(proj_dir))
        assert loaded.metadata["name"] == "Test"

    def test_snapshot(self, tmp_path):
        from prism.io.project import PRISMProject

        proj = PRISMProject.create(str(tmp_path / "snap_test"))
        proj.snapshot("Initial design", metrics={"rmsd": 0.5})

        assert len(proj.snapshots) == 1
        assert proj.snapshots[0].metrics["rmsd"] == 0.5

    def test_set_spec(self, tmp_path):
        from prism.core.cage import CageSpec, CavitySpec
        from prism.io.project import PRISMProject

        proj = PRISMProject.create(str(tmp_path / "spec_test"))
        spec = CageSpec(symmetry_group="I", cavity=CavitySpec(target_diameter_nm=15.0))
        proj.set_spec(spec)
        proj.save()

        loaded = PRISMProject.load(str(tmp_path / "spec_test"))
        assert loaded.spec["symmetry_group"] == "I"


class TestPDBIO:
    """Test PDB I/O."""

    def test_write_pdb(self, tmp_path):
        from prism.io.pdb import write_pdb

        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        path = write_pdb(coords, tmp_path / "test.pdb")

        assert path.exists()
        content = path.read_text()
        assert "ATOM" in content
        assert "END" in content

    def test_write_read_fasta(self, tmp_path):
        from prism.io.pdb import write_fasta, read_fasta

        seqs = {"protein_1": "ACDEFGHIKLMNPQRSTVWY", "protein_2": "MMMMMM"}
        path = write_fasta(seqs, tmp_path / "test.fasta")

        loaded = read_fasta(path)
        assert loaded["protein_1"] == "ACDEFGHIKLMNPQRSTVWY"
        assert loaded["protein_2"] == "MMMMMM"
