"""
Cavity analysis — characterise the interior void of a protein cage.

Wraps the Rust Monte Carlo cavity volume computation and provides
higher-level analysis: inscribed/circumscribed radius, surface area,
interior residue mapping, and volume comparison to target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Try Rust core
try:
    from prism._rust_core import compute_cavity_volume as _rs_cavity_volume
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass
class CavityReport:
    """Results of cavity analysis.

    Attributes
    ----------
    volume_angstrom3 : float
        Cavity volume in ų.
    volume_nm3 : float
        Cavity volume in nm³.
    inscribed_radius : float
        Minimum distance from cavity centre to protein wall (Å).
    effective_diameter_nm : float
        Effective cavity diameter in nm (2 × inscribed_radius / 10).
    void_fraction : float
        Fraction of the bounding sphere that is void.
    n_interior_residues : int
        Number of residues facing the interior.
    interior_residue_indices : NDArray
        Indices of interior-facing residues.
    target_diameter_nm : float or None
        Target cavity diameter from the cage spec.
    size_match_percent : float or None
        How close the actual cavity is to the target (%).
    """

    volume_angstrom3: float
    volume_nm3: float
    inscribed_radius: float
    effective_diameter_nm: float
    void_fraction: float
    n_interior_residues: int = 0
    interior_residue_indices: Optional[NDArray] = None
    target_diameter_nm: Optional[float] = None
    size_match_percent: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"Cavity Analysis Report",
            f"  Volume:    {self.volume_angstrom3:,.0f} ų ({self.volume_nm3:.1f} nm³)",
            f"  Diameter:  {self.effective_diameter_nm:.1f} nm (inscribed radius: {self.inscribed_radius:.1f} Å)",
            f"  Void frac: {self.void_fraction:.2%}",
            f"  Interior residues: {self.n_interior_residues}",
        ]
        if self.target_diameter_nm is not None:
            lines.append(f"  Target:    {self.target_diameter_nm:.1f} nm")
            lines.append(f"  Match:     {self.size_match_percent:.1f}%")
        return "\n".join(lines)


def analyse_cavity(
    coords: NDArray,
    elements: Optional[NDArray] = None,
    center: NDArray | list | tuple = (0.0, 0.0, 0.0),
    cage_radius: Optional[float] = None,
    n_samples: int = 500_000,
    target_diameter_nm: Optional[float] = None,
    ca_coords: Optional[NDArray] = None,
    cb_coords: Optional[NDArray] = None,
) -> CavityReport:
    """Perform full cavity analysis on a protein cage assembly.

    Parameters
    ----------
    coords : NDArray
        Shape (N, 3) — all atom coordinates of the full cage.
    elements : NDArray, optional
        Shape (N,) atomic numbers. If None, all atoms treated as carbon.
    center : array-like
        Cage centre. Default (0, 0, 0).
    cage_radius : float, optional
        Bounding radius for MC sampling. If None, inferred from coords.
    n_samples : int
        Number of Monte Carlo samples for volume estimation.
    target_diameter_nm : float, optional
        Target cavity diameter for match scoring.
    ca_coords : NDArray, optional
        Cα coords for interior residue identification.
    cb_coords : NDArray, optional
        Cβ coords for interior residue identification.

    Returns
    -------
    CavityReport
        Comprehensive cavity characterisation.
    """
    coords = np.asarray(coords, dtype=np.float64)
    center_arr = np.asarray(center, dtype=np.float64).ravel()

    if elements is None:
        elements = np.full(coords.shape[0], 6, dtype=np.uint8)  # all carbon

    if cage_radius is None:
        dists = np.linalg.norm(coords - center_arr, axis=1)
        cage_radius = float(dists.max() * 0.9)  # slightly inside the outer shell

    # Compute cavity volume
    if _HAS_RUST:
        result = _rs_cavity_volume(coords, elements, list(center_arr), cage_radius, n_samples)
        volume = result["volume_angstrom3"]
        inscribed_r = result["inscribed_radius"]
        void_frac = result["void_fraction"]
    else:
        volume, inscribed_r, void_frac = _cavity_volume_python(
            coords, elements, center_arr, cage_radius, n_samples
        )

    volume_nm3 = volume / 1000.0  # 1 nm³ = 1000 ų
    eff_diameter_nm = 2.0 * inscribed_r / 10.0

    # Interior residue detection
    n_interior = 0
    interior_indices = None
    if ca_coords is not None:
        from prism.design.surface_chem import select_interior_residues
        if cb_coords is None:
            cb_coords = ca_coords
        interior_indices = select_interior_residues(
            ca_coords, cb_coords, center=center_arr, max_distance=cage_radius
        )
        n_interior = len(interior_indices)

    # Size match
    size_match = None
    if target_diameter_nm is not None and eff_diameter_nm > 0:
        size_match = 100.0 * min(eff_diameter_nm, target_diameter_nm) / max(eff_diameter_nm, target_diameter_nm)

    return CavityReport(
        volume_angstrom3=volume,
        volume_nm3=volume_nm3,
        inscribed_radius=inscribed_r,
        effective_diameter_nm=eff_diameter_nm,
        void_fraction=void_frac,
        n_interior_residues=n_interior,
        interior_residue_indices=interior_indices,
        target_diameter_nm=target_diameter_nm,
        size_match_percent=size_match,
    )


def _cavity_volume_python(
    coords: NDArray, elements: NDArray, center: NDArray,
    cage_radius: float, n_samples: int,
) -> tuple[float, float, float]:
    """Pure Python Monte Carlo cavity volume (fallback)."""
    rng = np.random.default_rng(42)

    # VdW radii
    vdw = {6: 1.70, 7: 1.55, 8: 1.52, 16: 1.80, 1: 1.20}
    radii = np.array([vdw.get(int(e), 1.70) for e in elements])
    radii_sq = radii ** 2

    # Random points in bounding sphere
    # Use rejection sampling from cube
    n_in_sphere = 0
    n_in_cavity = 0
    min_wall_dist = float("inf")

    points = rng.uniform(-cage_radius, cage_radius, size=(n_samples, 3)) + center

    # Distance from centre
    dist_from_center = np.linalg.norm(points - center, axis=1)
    in_sphere = dist_from_center <= cage_radius

    for i in np.where(in_sphere)[0]:
        n_in_sphere += 1
        p = points[i]
        diffs = coords - p
        dists_sq = np.sum(diffs ** 2, axis=1)

        # Check if inside any atom
        inside_atom = np.any(dists_sq < radii_sq)

        if not inside_atom:
            n_in_cavity += 1
            min_d = float(np.sqrt(dists_sq.min()) - radii[dists_sq.argmin()])
            if min_d < min_wall_dist:
                min_wall_dist = min_d

    sphere_vol = 4.0 / 3.0 * np.pi * cage_radius ** 3
    void_fraction = n_in_cavity / n_in_sphere if n_in_sphere > 0 else 0.0
    volume = sphere_vol * void_fraction

    return volume, min_wall_dist, void_fraction
