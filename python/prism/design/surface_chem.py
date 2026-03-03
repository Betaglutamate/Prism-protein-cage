"""
Surface chemistry design — select and mutate interior-facing residues
for crystal-phase-selective nucleation.

This module bridges cavity analysis (which residues face the interior?)
with the residue_surface specification (what should those residues be?).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from prism.core.cage import CageDesign
from prism.core.residue_surface import SurfaceChemSpec, PHASE_RESIDUE_MAP

# Try importing Rust core for interior residue detection
try:
    from prism._rust_core import find_interior_residues as _rs_find_interior
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def select_interior_residues(
    ca_coords: NDArray,
    cb_coords: NDArray,
    center: NDArray | list[float] = (0.0, 0.0, 0.0),
    max_distance: Optional[float] = None,
) -> NDArray:
    """Identify residues whose side chains face the cage interior.

    A residue is interior-facing if the Cα→Cβ vector points toward
    the cage centre.

    Parameters
    ----------
    ca_coords : NDArray
        Shape (R, 3) — Cα coordinates.
    cb_coords : NDArray
        Shape (R, 3) — Cβ coordinates (Cα for glycine).
    center : array-like
        Cage centre coordinates. Default origin.
    max_distance : float, optional
        Maximum Cα-to-centre distance. If None, set to 80% of the
        maximum observed Cα distance.

    Returns
    -------
    NDArray
        1D array of residue indices (0-based) that face the interior.
    """
    ca = np.asarray(ca_coords, dtype=np.float64)
    cb = np.asarray(cb_coords, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).ravel()

    if max_distance is None:
        dists = np.linalg.norm(ca - center, axis=1)
        max_distance = 0.8 * dists.max()

    if _HAS_RUST:
        return _rs_find_interior(ca, cb, list(center), max_distance)
    else:
        # Pure Python fallback
        to_center = center - ca  # (R, 3)
        ca_cb = cb - ca           # (R, 3)
        dists = np.linalg.norm(to_center, axis=1)

        # Side chain points toward centre: positive dot product
        dot = np.sum(to_center * ca_cb, axis=1)

        mask = (dot > 0) & (dists < max_distance)
        return np.where(mask)[0].astype(np.uint32)


def design_surface_chemistry(
    cage: CageDesign,
    target_phase: str,
    ca_coords: Optional[NDArray] = None,
    cb_coords: Optional[NDArray] = None,
    assignment_pattern: str = "cluster",
) -> SurfaceChemSpec:
    """Full surface chemistry design: find interior residues and assign nucleation types.

    Parameters
    ----------
    cage : CageDesign
        The cage design (must have subunit_coords set).
    target_phase : str
        Target nanocrystal phase (e.g. "Fe3O4_magnetite", "Fe16N2").
    ca_coords : NDArray, optional
        Cα coordinates. If None, uses cage subunit coords (assumes all-atom).
    cb_coords : NDArray, optional
        Cβ coordinates. If None, uses Cα as proxy.
    assignment_pattern : str
        How to assign residue types: "uniform", "alternating", or "cluster".

    Returns
    -------
    SurfaceChemSpec
        Configured surface chemistry specification with assigned residues.
    """
    if target_phase not in PHASE_RESIDUE_MAP:
        known = ", ".join(PHASE_RESIDUE_MAP.keys())
        raise ValueError(f"Unknown phase '{target_phase}'. Known: {known}")

    # Use provided coords or infer from cage
    if ca_coords is None:
        if cage.subunit_coords is None:
            raise RuntimeError("Cage has no subunit_coords and no ca_coords provided.")
        # Assume Cα is every 4th atom (rough heuristic for all-atom PDB)
        # In practice, the user should provide extracted Cα coords
        ca_coords = cage.subunit_coords

    if cb_coords is None:
        cb_coords = ca_coords  # Glycine-like fallback

    # Find interior residues
    center = np.zeros(3)  # Cage centre at origin
    interior_indices = select_interior_residues(
        ca_coords, cb_coords, center=center,
        max_distance=cage.spec.cavity.target_radius_angstrom,
    )

    if len(interior_indices) == 0:
        raise RuntimeError(
            f"No interior-facing residues found within "
            f"{cage.spec.cavity.target_radius_angstrom:.0f} Å of cage centre. "
            f"Check subunit coordinates and cavity spec."
        )

    # Create and configure surface chemistry
    spec = SurfaceChemSpec.for_phase(target_phase)
    spec.assign_residues(interior_indices, pattern=assignment_pattern)

    # Attach to cage
    cage.surface_chem = spec
    cage.spec.surface_chemistry = spec.to_dict()

    return spec


def score_nucleation_potential(spec: SurfaceChemSpec) -> dict:
    """Score the nucleation potential of a surface chemistry assignment.

    Evaluates spatial distribution, coordination geometry compatibility,
    and residue type diversity.

    Returns
    -------
    dict
        Detailed scoring breakdown.
    """
    base_score = spec.score()

    # Additional heuristics
    n = base_score["n_assigned"]
    preferred = set(spec.preferred_residues)

    # Check diversity of assigned types
    assigned_types = set(spec.nucleation_residues.values())
    diversity = len(assigned_types) / len(preferred) if preferred else 0.0

    # Minimum cluster size check (need ≥3 coordinating residues for metal binding)
    has_minimum_cluster = n >= 3

    base_score.update({
        "diversity": diversity,
        "has_minimum_cluster": has_minimum_cluster,
        "assigned_types": sorted(assigned_types),
        "overall_score": (
            base_score["fraction_correct"] * 0.4
            + diversity * 0.3
            + (1.0 if has_minimum_cluster else 0.0) * 0.3
        ),
    })

    return base_score
