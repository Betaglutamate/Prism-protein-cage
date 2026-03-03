"""
CageDesign — the central data model for PRISM protein cage design.

Represents a complete cage specification including symmetry, cavity geometry,
interior surface chemistry, exterior interfaces, and structural data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from prism.core.symmetry import SymmetryGroup, SymmetryGroupName
from prism.core.residue_surface import SurfaceChemSpec


# ── Pydantic models for serialisable specifications ─────────────────


class CavitySpec(BaseModel):
    """Specification for the cage interior cavity."""

    target_diameter_nm: float = Field(
        ..., gt=0, le=50,
        description="Target cavity diameter in nanometres (5–20 nm typical).",
    )
    crystal_phase: Optional[str] = Field(
        default=None,
        description="Target crystal phase (e.g. 'Fe3O4_magnetite', 'Fe16N2').",
    )
    shape: Literal["spherical", "polyhedral"] = Field(
        default="spherical",
        description="Desired cavity shape.",
    )

    @property
    def target_radius_angstrom(self) -> float:
        """Target cavity radius in Angstroms."""
        return self.target_diameter_nm * 10.0 / 2.0

    @property
    def target_volume_angstrom3(self) -> float:
        """Approximate target volume in ų (assuming spherical)."""
        r = self.target_radius_angstrom
        return (4.0 / 3.0) * np.pi * r**3


class InterfaceSpec(BaseModel):
    """Specification for an exterior docking interface."""

    interface_id: str = Field(
        ..., description="Unique identifier for this interface type.",
    )
    partner_cage_id: Optional[str] = Field(
        default=None,
        description="ID of the partner cage (None = self-complementary).",
    )
    target_kd_nm: float = Field(
        default=100.0, gt=0,
        description="Target dissociation constant (nM).",
    )
    symmetry_copies: int = Field(
        default=1, ge=1,
        description="Number of symmetry-related copies of this interface per cage.",
    )
    hotspot_residues: list[int] = Field(
        default_factory=list,
        description="Residue indices on the cage exterior that define this interface patch.",
    )


class CageSpec(BaseModel):
    """Full specification for a protein cage design.

    This is the serialisable, JSON-compatible specification that defines
    exactly what cage to build. It drives the RFdiffusion and BindCraft
    design pipeline.
    """

    name: str = Field(default="unnamed_cage", description="Human-readable cage name.")
    symmetry_group: SymmetryGroupName = Field(
        ..., description="Symmetry group: T, O, or I.",
    )
    n_subunits: Optional[int] = Field(
        default=None,
        description="Number of subunits. If None, defaults to group order.",
    )
    subunit_length_range: tuple[int, int] = Field(
        default=(80, 150),
        description="(min, max) residues per subunit for RFdiffusion.",
    )
    cavity: CavitySpec = Field(
        ..., description="Interior cavity specification.",
    )
    surface_chemistry: Optional[dict] = Field(
        default=None,
        description="Serialised SurfaceChemSpec (call .to_dict()).",
    )
    exterior_interfaces: list[InterfaceSpec] = Field(
        default_factory=list,
        description="Exterior docking interface specifications.",
    )
    lattice_type: Optional[str] = Field(
        default=None,
        description="Target lattice type: 'sc', 'bcc', 'fcc', or 'custom'.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary metadata (design iteration, notes, etc.).",
    )

    @property
    def expected_n_subunits(self) -> int:
        """Number of subunits (from spec or group order)."""
        if self.n_subunits is not None:
            return self.n_subunits
        orders = {"T": 12, "O": 24, "I": 60}
        return orders[self.symmetry_group]


# ── CageDesign: the runtime object combining spec + structure ───────


@dataclass
class CageDesign:
    """A complete protein cage design, combining specification with structural data.

    This is the primary object users interact with. It holds the cage
    specification, the symmetry group, and (optionally) the designed
    subunit structure and its symmetry-expanded assembly.

    Examples
    --------
    >>> cage = CageDesign(
    ...     spec=CageSpec(
    ...         name="Fe16N2_cage",
    ...         symmetry_group="I",
    ...         cavity=CavitySpec(target_diameter_nm=10.0),
    ...     )
    ... )
    >>> cage.symmetry.order
    60
    >>> cage.spec.cavity.target_radius_angstrom
    50.0
    """

    spec: CageSpec
    symmetry: SymmetryGroup = field(init=False, repr=False)
    subunit_coords: Optional[NDArray] = field(default=None, repr=False)
    assembly_coords: Optional[NDArray] = field(default=None, repr=False)
    surface_chem: Optional[SurfaceChemSpec] = field(default=None, repr=False)

    def __post_init__(self):
        self.symmetry = SymmetryGroup.from_name(self.spec.symmetry_group)

        # Restore surface chemistry from spec if present
        if self.spec.surface_chemistry is not None and self.surface_chem is None:
            self.surface_chem = SurfaceChemSpec.from_dict(self.spec.surface_chemistry)

    # ── Assembly ────────────────────────────────────────────────────

    def set_subunit(self, coords: NDArray) -> None:
        """Set the subunit coordinates and expand the full cage.

        Parameters
        ----------
        coords : NDArray
            Shape (N, 3) coordinates of the asymmetric unit (one subunit).
        """
        self.subunit_coords = np.asarray(coords, dtype=np.float64)
        self.assembly_coords = self.symmetry.expand_coords(self.subunit_coords)

    def expand(self) -> NDArray:
        """Return the full cage coordinates by symmetry expansion.

        If subunit_coords is set, expands it. Otherwise raises an error.
        """
        if self.subunit_coords is None:
            raise RuntimeError("No subunit coordinates set. Call .set_subunit() first.")
        if self.assembly_coords is None:
            self.assembly_coords = self.symmetry.expand_coords(self.subunit_coords)
        return self.assembly_coords

    @property
    def n_atoms_per_subunit(self) -> Optional[int]:
        if self.subunit_coords is not None:
            return self.subunit_coords.shape[0]
        return None

    @property
    def n_atoms_total(self) -> Optional[int]:
        if self.assembly_coords is not None:
            return self.assembly_coords.shape[0]
        return None

    # ── Surface chemistry ───────────────────────────────────────────

    def set_surface_chemistry(self, target_phase: str) -> SurfaceChemSpec:
        """Configure interior surface chemistry for a target crystal phase.

        Parameters
        ----------
        target_phase : str
            Crystal phase name (e.g. "Fe3O4_magnetite", "Fe16N2").

        Returns
        -------
        SurfaceChemSpec
            The configured surface chemistry specification.
        """
        self.surface_chem = SurfaceChemSpec.for_phase(target_phase)
        self.spec.surface_chemistry = self.surface_chem.to_dict()
        return self.surface_chem

    # ── Serialisation ───────────────────────────────────────────────

    def save(self, directory: str | Path) -> Path:
        """Save the cage design to a directory.

        Creates:
        - spec.json — the CageSpec
        - subunit.npy — subunit coordinates (if set)
        - assembly.npy — assembly coordinates (if set)
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        spec_path = d / "spec.json"
        spec_path.write_text(self.spec.model_dump_json(indent=2))

        if self.subunit_coords is not None:
            np.save(d / "subunit.npy", self.subunit_coords)
        if self.assembly_coords is not None:
            np.save(d / "assembly.npy", self.assembly_coords)

        return d

    @classmethod
    def load(cls, directory: str | Path) -> CageDesign:
        """Load a cage design from a directory."""
        d = Path(directory)
        spec = CageSpec.model_validate_json((d / "spec.json").read_text())

        cage = cls(spec=spec)

        subunit_path = d / "subunit.npy"
        if subunit_path.exists():
            cage.subunit_coords = np.load(subunit_path)

        assembly_path = d / "assembly.npy"
        if assembly_path.exists():
            cage.assembly_coords = np.load(assembly_path)

        return cage

    # ── Display ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"CageDesign: {self.spec.name}",
            f"  Symmetry: {self.spec.symmetry_group} ({self.symmetry.order} operations)",
            f"  Subunits: {self.spec.expected_n_subunits}",
            f"  Cavity:   {self.spec.cavity.target_diameter_nm:.1f} nm diameter "
            f"({self.spec.cavity.shape})",
            f"  Subunit length: {self.spec.subunit_length_range[0]}–"
            f"{self.spec.subunit_length_range[1]} residues",
        ]
        if self.surface_chem:
            lines.append(f"  Phase:    {self.surface_chem.target_phase}")
            lines.append(f"  Interior residues assigned: {len(self.surface_chem.nucleation_residues)}")
        if self.subunit_coords is not None:
            lines.append(f"  Subunit atoms: {self.n_atoms_per_subunit}")
            lines.append(f"  Total atoms:   {self.n_atoms_total}")
        if self.spec.exterior_interfaces:
            lines.append(f"  Interfaces: {len(self.spec.exterior_interfaces)}")
        if self.spec.lattice_type:
            lines.append(f"  Lattice:   {self.spec.lattice_type}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CageDesign(name={self.spec.name!r}, "
            f"sym={self.spec.symmetry_group}, "
            f"cavity={self.spec.cavity.target_diameter_nm}nm)"
        )
