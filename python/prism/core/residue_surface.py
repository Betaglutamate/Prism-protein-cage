"""
Surface chemistry specification for nanocrystal nucleation.

Defines which residues face the cage interior and what amino acid
types should be placed there to template a specific crystal phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ── Crystal phase → preferred coordinating residues ─────────────────

PHASE_RESIDUE_MAP: dict[str, dict] = {
    "Fe3O4_magnetite": {
        "description": "Magnetite — iron oxide, ferrimagnetic",
        "coordinating_residues": ["HIS", "GLU", "ASP", "CYS"],
        "coordination_geometry": "octahedral",
        "metal_ions": ["Fe2+", "Fe3+"],
        "notes": "Inspired by ferritin; His/Glu clusters bind Fe ions for nucleation.",
    },
    "Fe16N2": {
        "description": "Iron nitride — rare-earth-free permanent magnet",
        "coordinating_residues": ["HIS", "ASN", "GLN", "GLU"],
        "coordination_geometry": "body-centred-tetragonal",
        "metal_ions": ["Fe2+"],
        "notes": (
            "Metastable phase; requires low-temperature nucleation. "
            "His for Fe binding, Asn/Gln for nitrogen coordination."
        ),
    },
    "CdSe_wurtzite": {
        "description": "Cadmium selenide — quantum dots",
        "coordinating_residues": ["CYS", "HIS", "MET"],
        "coordination_geometry": "tetrahedral",
        "metal_ions": ["Cd2+"],
        "notes": "Cys thiolate for Cd binding; inspired by metallothionein.",
    },
    "Pt_fcc": {
        "description": "Platinum — catalysis nanoparticles",
        "coordinating_residues": ["CYS", "MET", "HIS"],
        "coordination_geometry": "fcc",
        "metal_ions": ["Pt2+"],
        "notes": "Thiol/thioether coordination for Pt reduction/nucleation.",
    },
    "SiO2_amorphous": {
        "description": "Silica — structural/metamaterial applications",
        "coordinating_residues": ["SER", "THR", "LYS", "ARG"],
        "coordination_geometry": "amorphous",
        "metal_ions": [],
        "notes": "Inspired by diatom silaffins; Ser/Lys promote silicic acid condensation.",
    },
}


# Short amino acid name → single letter
AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

AA1_TO_3 = {v: k for k, v in AA3_TO_1.items()}


@dataclass
class SurfaceChemSpec:
    """Specification for interior surface chemistry.

    Parameters
    ----------
    target_phase : str
        Target nanocrystal phase (key in PHASE_RESIDUE_MAP).
    nucleation_residues : dict[int, str]
        Mapping of residue index → target 3-letter amino acid name.
        These residues will be mutated to present the correct functional
        groups for crystal nucleation.
    coordination_geometry : str
        Expected coordination geometry at nucleation sites.
    notes : str
        Design rationale or additional notes.
    """

    target_phase: str
    nucleation_residues: dict[int, str] = field(default_factory=dict)
    coordination_geometry: str = ""
    notes: str = ""

    @classmethod
    def for_phase(cls, phase: str) -> SurfaceChemSpec:
        """Create a SurfaceChemSpec template for a known crystal phase.

        Parameters
        ----------
        phase : str
            Crystal phase name (e.g. "Fe3O4_magnetite", "Fe16N2").
        """
        if phase not in PHASE_RESIDUE_MAP:
            known = ", ".join(PHASE_RESIDUE_MAP.keys())
            raise ValueError(f"Unknown phase '{phase}'. Known phases: {known}")

        info = PHASE_RESIDUE_MAP[phase]
        return cls(
            target_phase=phase,
            coordination_geometry=info["coordination_geometry"],
            notes=info["notes"],
        )

    @property
    def preferred_residues(self) -> list[str]:
        """Return the preferred coordinating residue types for this phase."""
        if self.target_phase in PHASE_RESIDUE_MAP:
            return PHASE_RESIDUE_MAP[self.target_phase]["coordinating_residues"]
        return []

    def assign_residues(
        self,
        interior_indices: NDArray | list[int],
        pattern: str = "alternating",
    ) -> SurfaceChemSpec:
        """Auto-assign nucleation residues to interior-facing positions.

        Parameters
        ----------
        interior_indices : array-like
            Residue indices that face the cage interior.
        pattern : str
            Assignment pattern:
            - "alternating": cycle through preferred residue types.
            - "uniform": use the first preferred residue for all.
            - "cluster": group residues into metal-binding clusters.
        """
        preferred = self.preferred_residues
        if not preferred:
            raise ValueError(f"No preferred residues known for phase '{self.target_phase}'")

        indices = list(interior_indices)
        assignments = {}

        if pattern == "uniform":
            for idx in indices:
                assignments[int(idx)] = preferred[0]
        elif pattern == "alternating":
            for i, idx in enumerate(indices):
                assignments[int(idx)] = preferred[i % len(preferred)]
        elif pattern == "cluster":
            # Group into clusters of 3-4 residues and assign one type per cluster
            cluster_size = min(4, len(preferred))
            for i, idx in enumerate(indices):
                cluster_pos = i % cluster_size
                assignments[int(idx)] = preferred[cluster_pos]
        else:
            raise ValueError(f"Unknown pattern '{pattern}'. Use 'uniform', 'alternating', or 'cluster'.")

        self.nucleation_residues = assignments
        return self

    def score(self) -> dict:
        """Heuristic quality assessment of the surface chemistry assignment.

        Returns
        -------
        dict
            Scores and diagnostic information.
        """
        n_assigned = len(self.nucleation_residues)
        preferred = set(self.preferred_residues)
        n_correct_type = sum(1 for aa in self.nucleation_residues.values() if aa in preferred)

        return {
            "n_assigned": n_assigned,
            "n_correct_type": n_correct_type,
            "fraction_correct": n_correct_type / n_assigned if n_assigned > 0 else 0.0,
            "target_phase": self.target_phase,
            "coordination_geometry": self.coordination_geometry,
        }

    def to_dict(self) -> dict:
        """Serialise to dictionary."""
        return {
            "target_phase": self.target_phase,
            "nucleation_residues": {str(k): v for k, v in self.nucleation_residues.items()},
            "coordination_geometry": self.coordination_geometry,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SurfaceChemSpec:
        """Deserialise from dictionary."""
        return cls(
            target_phase=d["target_phase"],
            nucleation_residues={int(k): v for k, v in d.get("nucleation_residues", {}).items()},
            coordination_geometry=d.get("coordination_geometry", ""),
            notes=d.get("notes", ""),
        )
