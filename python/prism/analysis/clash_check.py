"""
Steric clash detection for cage assemblies and lattice packing.

Thin wrappers around the Rust KD-tree, plus helpers to interpret
and report clashes found during design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Try Rust core
try:
    from prism._rust_core import clash_check as _rs_clash_check, find_contacts as _rs_contacts
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass
class ClashReport:
    """Summary of steric clashes between two coordinate sets.

    Attributes
    ----------
    n_clashes : int
        Total atom-atom clashes below threshold.
    worst_overlap : float
        Smallest inter-atomic distance found (Å).
    clash_pairs : list[tuple[int, int]]
        Indices of clashing atom pairs (capped for large sets).
    passed : bool
        True if no clashes found.
    cutoff : float
        Distance cutoff used (Å).
    """

    n_clashes: int
    worst_overlap: float
    clash_pairs: list[tuple[int, int]] = field(default_factory=list)
    passed: bool = True
    cutoff: float = 2.0

    def summary(self) -> str:
        status = "PASS ✓" if self.passed else f"FAIL ✗ ({self.n_clashes} clashes)"
        lines = [
            f"Clash Report: {status}",
            f"  Cutoff: {self.cutoff:.1f} Å",
        ]
        if not self.passed:
            lines.append(f"  Worst overlap: {self.worst_overlap:.2f} Å")
            if self.clash_pairs:
                lines.append(f"  First 5 clash pairs: {self.clash_pairs[:5]}")
        return "\n".join(lines)


def check_clashes(
    coords_a: NDArray,
    coords_b: NDArray,
    cutoff: float = 2.0,
    max_pairs: int = 1000,
) -> ClashReport:
    """Detect steric clashes between two sets of atoms.

    Parameters
    ----------
    coords_a : NDArray
        Shape (N, 3) reference structure.
    coords_b : NDArray
        Shape (M, 3) query structure.
    cutoff : float
        Clash distance threshold in Å (default 2.0).
    max_pairs : int
        Maximum number of clash pairs to record.

    Returns
    -------
    ClashReport
    """
    coords_a = np.ascontiguousarray(coords_a, dtype=np.float64)
    coords_b = np.ascontiguousarray(coords_b, dtype=np.float64)

    if _HAS_RUST:
        result = _rs_clash_check(coords_a, coords_b, cutoff)
        n_clashes = result["n_clashes"]
        worst = result["min_distance"]
        pairs = [(int(a), int(b)) for a, b in result.get("pairs", [])[:max_pairs]]
    else:
        n_clashes, worst, pairs = _clash_check_python(coords_a, coords_b, cutoff, max_pairs)

    return ClashReport(
        n_clashes=n_clashes,
        worst_overlap=worst,
        clash_pairs=pairs,
        passed=(n_clashes == 0),
        cutoff=cutoff,
    )


def check_self_clashes(
    coords: NDArray,
    chain_ids: NDArray,
    cutoff: float = 2.0,
    max_pairs: int = 500,
) -> ClashReport:
    """Detect inter-chain clashes within an assembly.

    Intra-chain contacts are excluded.

    Parameters
    ----------
    coords : NDArray
        Shape (N, 3) assembly coordinates.
    chain_ids : NDArray
        Shape (N,) chain identifiers for each atom.
    cutoff : float
        Clash distance threshold (Å).

    Returns
    -------
    ClashReport
    """
    unique_chains = np.unique(chain_ids)
    all_clashes = 0
    worst = float("inf")
    pairs: list[tuple[int, int]] = []

    for i, ci in enumerate(unique_chains):
        mask_i = chain_ids == ci
        coords_i = coords[mask_i]
        offset_i = np.where(mask_i)[0]

        for cj in unique_chains[i + 1 :]:
            mask_j = chain_ids == cj
            coords_j = coords[mask_j]
            offset_j = np.where(mask_j)[0]

            report = check_clashes(coords_i, coords_j, cutoff, max_pairs=50)

            all_clashes += report.n_clashes
            if report.worst_overlap < worst:
                worst = report.worst_overlap

            for a, b in report.clash_pairs:
                pairs.append((int(offset_i[a]), int(offset_j[b])))
                if len(pairs) >= max_pairs:
                    break

    if worst == float("inf"):
        worst = cutoff + 1.0  # No contacts at all

    return ClashReport(
        n_clashes=all_clashes,
        worst_overlap=worst,
        clash_pairs=pairs[:max_pairs],
        passed=(all_clashes == 0),
        cutoff=cutoff,
    )


def find_contacts(
    coords_a: NDArray,
    coords_b: NDArray,
    cutoff: float = 4.5,
) -> list[tuple[int, int, float]]:
    """Find inter-atomic contacts within a distance cutoff.

    Parameters
    ----------
    coords_a, coords_b : NDArray
        Shape (N, 3) coordinate arrays.
    cutoff : float
        Contact distance threshold in Å.

    Returns
    -------
    list of (i, j, distance) tuples
    """
    coords_a = np.ascontiguousarray(coords_a, dtype=np.float64)
    coords_b = np.ascontiguousarray(coords_b, dtype=np.float64)

    if _HAS_RUST:
        result = _rs_contacts(coords_a, coords_b, cutoff)
        return [(int(r[0]), int(r[1]), float(r[2])) for r in result]
    else:
        return _find_contacts_python(coords_a, coords_b, cutoff)


# ── Pure Python fallbacks ─────────────────────────────────────────────

def _clash_check_python(
    a: NDArray, b: NDArray, cutoff: float, max_pairs: int,
) -> tuple[int, float, list[tuple[int, int]]]:
    """Brute-force clash detection (O(N×M))."""
    from scipy.spatial import cKDTree

    tree = cKDTree(a)
    results = tree.query_ball_point(b, cutoff)

    n_clashes = 0
    worst = float("inf")
    pairs: list[tuple[int, int]] = []

    for j, neighbours in enumerate(results):
        for i_a in neighbours:
            d = float(np.linalg.norm(a[i_a] - b[j]))
            if d < cutoff:
                n_clashes += 1
                if d < worst:
                    worst = d
                if len(pairs) < max_pairs:
                    pairs.append((i_a, j))

    if worst == float("inf"):
        worst = cutoff + 1.0

    return n_clashes, worst, pairs


def _find_contacts_python(
    a: NDArray, b: NDArray, cutoff: float,
) -> list[tuple[int, int, float]]:
    """Brute-force contact finding (O(N×M))."""
    from scipy.spatial import cKDTree

    tree = cKDTree(a)
    results = tree.query_ball_point(b, cutoff)

    contacts: list[tuple[int, int, float]] = []
    for j, neighbours in enumerate(results):
        for i_a in neighbours:
            d = float(np.linalg.norm(a[i_a] - b[j]))
            contacts.append((i_a, j, d))

    return contacts
