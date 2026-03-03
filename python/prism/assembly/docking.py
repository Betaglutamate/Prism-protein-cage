"""
Docking interface design orchestration.

Manages the workflow of identifying exterior surface patches on a cage,
designing orthogonal docking interfaces via BindCraft, and validating
interface quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from prism.core.cage import CageDesign, InterfaceSpec

# Try importing Rust core
try:
    from prism._rust_core import find_contacts as _rs_find_contacts
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass
class ExteriorPatch:
    """A patch on the cage exterior suitable for docking interface design.

    Attributes
    ----------
    residue_indices : list[int]
        Residue indices in the subunit that form this patch.
    center : NDArray
        Centroid of the patch (Å).
    normal : NDArray
        Outward-facing normal vector.
    area : float
        Approximate patch area (ų).
    symmetry_axis : str
        Which symmetry axis this patch is near (e.g. "C2", "C3", "C5").
    """

    residue_indices: list[int]
    center: NDArray
    normal: NDArray
    area: float = 0.0
    symmetry_axis: str = ""


def identify_exterior_patches(
    cage: CageDesign,
    ca_coords: NDArray,
    n_patches: int = 2,
    min_patch_size: int = 5,
) -> list[ExteriorPatch]:
    """Identify exterior surface patches suitable for docking interfaces.

    Finds regions on the cage exterior that are:
    1. Solvent-accessible (not buried in the cage interior)
    2. Near symmetry-related interfaces (C2, C3, etc.)
    3. Large enough for a viable binding interface

    Parameters
    ----------
    cage : CageDesign
        The cage design with subunit_coords set.
    ca_coords : NDArray
        Shape (R, 3) — Cα coordinates of one subunit.
    n_patches : int
        Number of distinct patches to identify. Default 2.
    min_patch_size : int
        Minimum number of residues per patch.

    Returns
    -------
    list[ExteriorPatch]
        Identified patches, sorted by area (largest first).
    """
    center = np.zeros(3)
    n_res = len(ca_coords)

    # Compute direction from centre to each residue
    directions = ca_coords - center
    distances = np.linalg.norm(directions, axis=1)
    directions_norm = directions / distances[:, None]

    # Exterior residues: those farther from centre than the median
    median_dist = np.median(distances)
    exterior_mask = distances > median_dist
    exterior_indices = np.where(exterior_mask)[0]

    if len(exterior_indices) < min_patch_size:
        return []

    # Simple clustering: group exterior residues by angular proximity
    # Use k-means-like approach on normalised direction vectors
    patches = []
    remaining = set(exterior_indices.tolist())

    for _ in range(n_patches):
        if len(remaining) < min_patch_size:
            break

        # Seed: the residue farthest from centre among remaining
        remaining_list = sorted(remaining)
        seed_idx = max(remaining_list, key=lambda i: distances[i])

        # Grow patch: add nearby residues (angular distance < threshold)
        seed_dir = directions_norm[seed_idx]
        patch_indices = []

        for idx in list(remaining):
            cos_angle = np.dot(directions_norm[idx], seed_dir)
            if cos_angle > 0.7:  # ~45° cone
                patch_indices.append(idx)

        if len(patch_indices) < min_patch_size:
            # Try a larger cone
            for idx in list(remaining):
                cos_angle = np.dot(directions_norm[idx], seed_dir)
                if cos_angle > 0.5 and idx not in patch_indices:  # ~60° cone
                    patch_indices.append(idx)

        if len(patch_indices) >= min_patch_size:
            patch_coords = ca_coords[patch_indices]
            patch_center = patch_coords.mean(axis=0)
            patch_normal = patch_center - center
            patch_normal = patch_normal / np.linalg.norm(patch_normal)

            # Approximate area from convex hull (simplified)
            area = float(len(patch_indices) * 50.0)  # rough: 50 ų per residue

            # Determine nearest symmetry axis
            sym_axis = _nearest_symmetry_axis(patch_normal, cage.symmetry)

            patches.append(ExteriorPatch(
                residue_indices=sorted(patch_indices),
                center=patch_center,
                normal=patch_normal,
                area=area,
                symmetry_axis=sym_axis,
            ))

            remaining -= set(patch_indices)

    # Sort by area (largest first)
    patches.sort(key=lambda p: p.area, reverse=True)
    return patches


def _nearest_symmetry_axis(normal: NDArray, symmetry) -> str:
    """Find which symmetry axis a normal vector is closest to."""
    axes = symmetry.get_symmetry_axes()
    best_axis = ""
    best_cos = 0.0

    for axis_type, axis_vectors in axes.items():
        for ax in axis_vectors:
            cos_val = abs(np.dot(normal, ax))
            if cos_val > best_cos:
                best_cos = cos_val
                best_axis = axis_type

    return best_axis


def design_orthogonal_interfaces(
    cage: CageDesign,
    patches: list[ExteriorPatch],
    *,
    target_lattice: str = "sc",
) -> list[InterfaceSpec]:
    """Generate InterfaceSpec objects for orthogonal docking interfaces.

    Each patch gets a unique interface specification designed to ensure
    orthogonality (non-cross-reactivity between different interface types).

    Parameters
    ----------
    cage : CageDesign
        The cage design.
    patches : list[ExteriorPatch]
        Exterior patches from identify_exterior_patches().
    target_lattice : str
        Target lattice type: "sc" (simple cubic), "bcc", "fcc", "hex".

    Returns
    -------
    list[InterfaceSpec]
        Interface specifications ready for BindCraft design.
    """
    # Lattice geometry determines how many interface types we need
    n_interfaces_needed = {
        "sc": 1,   # Simple cubic: 1 interface type, 6 contacts
        "bcc": 2,  # BCC: 2 types, 8+6 contacts
        "fcc": 1,  # FCC: 1 type, 12 contacts
        "hex": 2,  # Hexagonal: 2 types
    }

    n_needed = n_interfaces_needed.get(target_lattice, len(patches))
    interfaces = []

    for i, patch in enumerate(patches[:n_needed]):
        # Determine symmetry copies from the cage symmetry
        sym_copies = _symmetry_copies_for_axis(patch.symmetry_axis, cage.spec.symmetry_group)

        interface = InterfaceSpec(
            interface_id=f"IF_{i + 1}_{patch.symmetry_axis}",
            partner_cage_id=None,  # self-complementary
            target_kd_nm=100.0,
            symmetry_copies=sym_copies,
            hotspot_residues=patch.residue_indices,
        )
        interfaces.append(interface)

    # Attach to cage spec
    cage.spec.exterior_interfaces = interfaces
    cage.spec.lattice_type = target_lattice

    return interfaces


def _symmetry_copies_for_axis(axis_type: str, symmetry_group: str) -> int:
    """Number of symmetry-related copies of an interface on a given axis."""
    copies = {
        "T": {"C2": 3, "C3": 4},
        "O": {"C2": 6, "C3": 4, "C4": 3},
        "I": {"C2": 15, "C3": 10, "C5": 6},
    }
    return copies.get(symmetry_group, {}).get(axis_type, 1)
