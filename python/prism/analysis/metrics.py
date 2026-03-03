"""
Design quality metrics for PRISM protein cages.

Collects packing density, void fraction, interface BSA, symmetry RMSD,
and composite scoring into a single `QualityReport`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Try Rust core
try:
    from prism._rust_core import kabsch_rmsd as _rs_rmsd, compute_sasa as _rs_sasa
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass
class QualityReport:
    """Composite quality metrics for a cage design.

    Attributes
    ----------
    symmetry_rmsd : float
        RMSD between subunits after symmetry superposition (Å).
    packing_density : float
        Fraction of cage shell volume occupied by atoms.
    void_fraction : float
        Fraction of bounding sphere that is empty (from cavity analysis).
    interface_bsa : float
        Total buried surface area at inter-subunit interfaces (Ų).
    per_interface_bsa : list[float]
        BSA per symmetry-unique interface.
    sasa_monomer : float
        SASA of isolated subunit (Ų).
    sasa_complex : float
        SASA of the full assembled cage (Ų).
    n_interface_contacts : int
        Number of inter-chain contacts at interfaces.
    composite_score : float
        Weighted composite quality score (0-100).
    """

    symmetry_rmsd: float = 0.0
    packing_density: float = 0.0
    void_fraction: float = 0.0
    interface_bsa: float = 0.0
    per_interface_bsa: list[float] | None = None
    sasa_monomer: float = 0.0
    sasa_complex: float = 0.0
    n_interface_contacts: int = 0
    composite_score: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Quality Report  (score: {self.composite_score:.1f}/100)",
            f"  Symmetry RMSD:   {self.symmetry_rmsd:.2f} Å",
            f"  Packing density: {self.packing_density:.2%}",
            f"  Void fraction:   {self.void_fraction:.2%}",
            f"  Interface BSA:   {self.interface_bsa:,.0f} Ų",
            f"  Contacts:        {self.n_interface_contacts}",
            f"  SASA monomer:    {self.sasa_monomer:,.0f} Ų",
            f"  SASA complex:    {self.sasa_complex:,.0f} Ų",
        ]
        return "\n".join(lines)


# ── Individual metrics ────────────────────────────────────────────────

def symmetry_rmsd(
    subunit_coords: list[NDArray],
    reference_idx: int = 0,
) -> float:
    """Compute mean pairwise RMSD between symmetry-related subunits.

    All subunits are superposed onto `reference_idx` and the mean
    RMSD is returned.

    Parameters
    ----------
    subunit_coords : list of NDArray
        Each element is shape (N, 3) — Cα coordinates of one subunit.
    reference_idx : int
        Index of the reference subunit.

    Returns
    -------
    float
        Mean RMSD in Å.
    """
    ref = np.ascontiguousarray(subunit_coords[reference_idx], dtype=np.float64)
    rmsds = []

    for i, sc in enumerate(subunit_coords):
        if i == reference_idx:
            continue
        coords = np.ascontiguousarray(sc, dtype=np.float64)
        if _HAS_RUST:
            rmsds.append(_rs_rmsd(ref, coords))
        else:
            rmsds.append(_rmsd_python(ref, coords))

    return float(np.mean(rmsds)) if rmsds else 0.0


def interface_bsa(
    coords: NDArray,
    chain_ids: NDArray,
    elements: Optional[NDArray] = None,
    probe_radius: float = 1.4,
) -> tuple[float, list[float]]:
    """Compute buried surface area at inter-chain interfaces.

    BSA = Σ_i SASA(chain i) - SASA(complex)

    Parameters
    ----------
    coords : NDArray
        Shape (N, 3) assembly coordinates.
    chain_ids : NDArray
        Shape (N,) chain identifiers.
    elements : NDArray, optional
        Atomic numbers. Default: all carbon.
    probe_radius : float
        Probe radius for SASA computation (Å).

    Returns
    -------
    total_bsa : float
        Total BSA in Ų.
    per_chain_bsa : list[float]
        BSA contribution of each chain.
    """
    if elements is None:
        elements = np.full(coords.shape[0], 6, dtype=np.uint8)

    coords = np.ascontiguousarray(coords, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.uint8)

    # SASA of entire complex
    sasa_complex = _compute_sasa(coords, elements, probe_radius)

    # SASA of each chain in isolation
    unique_chains = np.unique(chain_ids)
    per_chain_sasa = []
    per_chain_bsa_vals = []

    for ch in unique_chains:
        mask = chain_ids == ch
        ch_coords = np.ascontiguousarray(coords[mask], dtype=np.float64)
        ch_elems = elements[mask]
        per_chain_sasa.append(_compute_sasa(ch_coords, ch_elems, probe_radius))

    sum_individual = sum(per_chain_sasa)
    total_bsa = sum_individual - sasa_complex

    # Approximate per-chain contribution
    for s in per_chain_sasa:
        frac = s / sum_individual if sum_individual > 0 else 1.0 / len(unique_chains)
        per_chain_bsa_vals.append(total_bsa * frac)

    return total_bsa, per_chain_bsa_vals


def packing_density(
    coords: NDArray,
    elements: Optional[NDArray] = None,
    shell_outer_radius: Optional[float] = None,
    shell_inner_radius: Optional[float] = None,
) -> float:
    """Compute the packing density of atoms in the cage shell.

    packing = Σ(4/3 π r_vdw³) / shell_volume

    Parameters
    ----------
    coords : NDArray
        Shape (N, 3).
    elements : NDArray, optional
        Atomic numbers.
    shell_outer_radius : float, optional
        Outer radius of cage shell. Inferred from coords if None.
    shell_inner_radius : float, optional
        Inner radius. Inferred if None.

    Returns
    -------
    float
        Packing density fraction (0.0–1.0).
    """
    vdw = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 16: 1.80}

    if elements is None:
        elements = np.full(coords.shape[0], 6, dtype=np.uint8)

    # Atom volumes
    radii = np.array([vdw.get(int(e), 1.70) for e in elements])
    atom_volumes = (4.0 / 3.0) * np.pi * radii ** 3

    # Shell volume
    dists = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
    if shell_outer_radius is None:
        shell_outer_radius = float(dists.max())
    if shell_inner_radius is None:
        shell_inner_radius = float(max(dists.min(), shell_outer_radius * 0.3))

    shell_vol = (4.0 / 3.0) * np.pi * (shell_outer_radius ** 3 - shell_inner_radius ** 3)

    return float(atom_volumes.sum() / shell_vol) if shell_vol > 0 else 0.0


# ── Composite scoring ─────────────────────────────────────────────────

def score_design(
    subunit_coords: Optional[list[NDArray]] = None,
    assembly_coords: Optional[NDArray] = None,
    chain_ids: Optional[NDArray] = None,
    elements: Optional[NDArray] = None,
    void_fraction: float = 0.0,
    weights: Optional[dict[str, float]] = None,
) -> QualityReport:
    """Compute a composite quality score for a cage design.

    Default weights (0-100 scale):
      symmetry_rmsd:   30  —  lower is better (target < 0.5 Å)
      packing:         15  —  moderate packing is good (~0.2-0.4)
      bsa:             30  —  more BSA = better interfaces
      void_frac:       25  —  target range depends on design

    Returns
    -------
    QualityReport
    """
    default_weights = {
        "symmetry_rmsd": 30.0,
        "packing": 15.0,
        "bsa": 30.0,
        "void_fraction": 25.0,
    }
    w = {**default_weights, **(weights or {})}

    # Symmetry RMSD score
    sym_rmsd = 0.0
    if subunit_coords and len(subunit_coords) > 1:
        sym_rmsd = symmetry_rmsd(subunit_coords)
    sym_score = max(0.0, 1.0 - sym_rmsd / 2.0)  # 0 at 2 Å, 1 at 0 Å

    # Packing score
    pack = 0.0
    if assembly_coords is not None:
        pack = packing_density(assembly_coords, elements)
    # Ideal packing ~ 0.3 for protein cages
    pack_score = 1.0 - abs(pack - 0.3) / 0.3
    pack_score = max(0.0, min(1.0, pack_score))

    # Interface BSA score
    total_bsa = 0.0
    per_bsa: list[float] = []
    sasa_mono = 0.0
    sasa_cplx = 0.0
    n_contacts = 0
    if assembly_coords is not None and chain_ids is not None:
        total_bsa, per_bsa = interface_bsa(assembly_coords, chain_ids, elements)
        # SASA values
        if elements is None:
            elems = np.full(assembly_coords.shape[0], 6, dtype=np.uint8)
        else:
            elems = elements
        sasa_cplx = _compute_sasa(assembly_coords, elems)

        # Monomer SASA (first chain)
        mask0 = chain_ids == np.unique(chain_ids)[0]
        sasa_mono = _compute_sasa(
            np.ascontiguousarray(assembly_coords[mask0], dtype=np.float64),
            elems[mask0],
        )

    # BSA score — target ~1500 Ų per interface
    n_chains = len(np.unique(chain_ids)) if chain_ids is not None else 1
    bsa_per_interface = total_bsa / max(n_chains, 1)
    bsa_score = min(1.0, bsa_per_interface / 1500.0)

    # Void fraction score — higher void fraction = larger cavity
    void_score = min(1.0, void_fraction / 0.5) if void_fraction > 0 else 0.0

    # Composite
    composite = (
        w["symmetry_rmsd"] * sym_score
        + w["packing"] * pack_score
        + w["bsa"] * bsa_score
        + w["void_fraction"] * void_score
    )

    return QualityReport(
        symmetry_rmsd=sym_rmsd,
        packing_density=pack,
        void_fraction=void_fraction,
        interface_bsa=total_bsa,
        per_interface_bsa=per_bsa if per_bsa else None,
        sasa_monomer=sasa_mono,
        sasa_complex=sasa_cplx,
        n_interface_contacts=n_contacts,
        composite_score=composite,
    )


# ── Helpers ───────────────────────────────────────────────────────────

def _compute_sasa(coords: NDArray, elements: NDArray, probe_radius: float = 1.4) -> float:
    """Dispatch SASA computation to Rust or Python."""
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.uint8)

    if _HAS_RUST:
        return _rs_sasa(coords, elements, probe_radius)
    else:
        return _sasa_python(coords, elements, probe_radius)


def _sasa_python(coords: NDArray, elements: NDArray, probe_radius: float = 1.4) -> float:
    """Simplified Shrake-Rupley SASA (pure Python fallback)."""
    vdw = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 16: 1.80}
    radii = np.array([vdw.get(int(e), 1.70) + probe_radius for e in elements])

    n_points = 92  # golden spiral test points
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * np.arange(n_points)
    z = np.linspace(1.0 - 1.0 / n_points, -1.0 + 1.0 / n_points, n_points)
    r_xy = np.sqrt(1.0 - z * z)
    sphere_pts = np.column_stack([r_xy * np.cos(theta), r_xy * np.sin(theta), z])

    total_sasa = 0.0
    for i in range(len(coords)):
        test = coords[i] + radii[i] * sphere_pts
        # Check if test points are inside other atoms
        exposed = 0
        for p in test:
            dists_sq = np.sum((coords - p) ** 2, axis=1)
            radii_sq = radii ** 2
            inside = np.any(dists_sq[np.arange(len(coords)) != i] < radii_sq[np.arange(len(coords)) != i])
            if not inside:
                exposed += 1
        area_per_point = 4.0 * np.pi * radii[i] ** 2 / n_points
        total_sasa += exposed * area_per_point

    return total_sasa


def _rmsd_python(a: NDArray, b: NDArray) -> float:
    """Kabsch RMSD (pure Python fallback)."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)

    H = a.T @ b
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    b_rot = (R @ b.T).T

    return float(np.sqrt(np.mean(np.sum((a - b_rot) ** 2, axis=1))))
