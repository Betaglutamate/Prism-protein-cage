"""
Symmetry group operations for protein cage design.

Provides a Pythonic API wrapping the Rust geometry core for applying
polyhedral symmetry operations (T, O, I) to atomic coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


# Try importing the Rust core; fall back to pure-Python if not compiled
try:
    from prism._rust_core import (
        apply_symmetry_ops as _rs_apply_symmetry_ops,
        get_symmetry_operations as _rs_get_symmetry_operations,
        get_group_generators as _rs_get_group_generators,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


# ── Fallback pure-Python symmetry generators ────────────────────────────

_PHI = (1.0 + np.sqrt(5.0)) / 2.0


def _rotation_matrix(axis: NDArray, angle: float) -> NDArray:
    """Rodrigues' rotation formula."""
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    return np.array([
        [t * ux * ux + c,      t * ux * uy - s * uz, t * ux * uz + s * uy],
        [t * ux * uy + s * uz, t * uy * uy + c,      t * uy * uz - s * ux],
        [t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c],
    ])


def _generate_group(generators: list[NDArray], max_ops: int) -> list[NDArray]:
    """Close a set of generators under matrix multiplication."""
    group = [np.eye(3)]
    for g in generators:
        if not any(np.allclose(g, m) for m in group):
            group.append(g)

    changed = True
    while changed and len(group) < max_ops + 1:
        changed = False
        new_ops = []
        for a in list(group):
            for b in list(group):
                prod = a @ b
                if not any(np.allclose(prod, m) for m in group + new_ops):
                    new_ops.append(prod)
                    changed = True
        group.extend(new_ops)

    return group[:max_ops]


def _tetrahedral_ops_py() -> list[NDArray]:
    axis_c3 = np.array([1, 1, 1]) / np.sqrt(3)
    axis_c2 = np.array([1, 0, 0])
    c3 = _rotation_matrix(axis_c3, 2 * np.pi / 3)
    c2 = _rotation_matrix(axis_c2, np.pi)
    return _generate_group([c3, c2], 12)


def _octahedral_ops_py() -> list[NDArray]:
    c4 = _rotation_matrix(np.array([0, 0, 1]), np.pi / 2)
    c3 = _rotation_matrix(np.array([1, 1, 1]) / np.sqrt(3), 2 * np.pi / 3)
    return _generate_group([c4, c3], 24)


def _icosahedral_ops_py() -> list[NDArray]:
    axis_c5 = np.array([0, 1, _PHI])
    axis_c5 = axis_c5 / np.linalg.norm(axis_c5)
    c5 = _rotation_matrix(axis_c5, 2 * np.pi / 5)
    c3 = _rotation_matrix(np.array([1, 1, 1]) / np.sqrt(3), 2 * np.pi / 3)
    c2 = _rotation_matrix(np.array([0, 0, 1]), np.pi)
    return _generate_group([c5, c3, c2], 60)


# ── Symmetry axes information ───────────────────────────────────────────

_SYMMETRY_INFO = {
    "T": {
        "order": 12,
        "n_c2": 3, "n_c3": 4,
        "description": "Chiral tetrahedral point group (12 operations)",
    },
    "O": {
        "order": 24,
        "n_c2": 6, "n_c3": 4, "n_c4": 3,
        "description": "Chiral octahedral point group (24 operations)",
    },
    "I": {
        "order": 60,
        "n_c2": 15, "n_c3": 10, "n_c5": 6,
        "description": "Chiral icosahedral point group (60 operations)",
    },
}


SymmetryGroupName = Literal["T", "O", "I"]


@dataclass
class SymmetryGroup:
    """A polyhedral point-group symmetry for protein cage design.

    Attributes
    ----------
    name : str
        Group name: "T" (tetrahedral), "O" (octahedral), "I" (icosahedral).
    operations : NDArray
        Shape (M, 3, 3) — all rotation matrices in the group.
    order : int
        Number of group operations.
    """

    name: SymmetryGroupName
    operations: NDArray = field(repr=False)

    @classmethod
    def from_name(cls, name: SymmetryGroupName) -> SymmetryGroup:
        """Create a symmetry group from its name.

        Parameters
        ----------
        name : str
            "T" for tetrahedral (12 ops), "O" for octahedral (24 ops),
            "I" for icosahedral (60 ops).
        """
        if _HAS_RUST:
            stacked = _rs_get_symmetry_operations(name)  # (M*3, 3)
            n_ops = stacked.shape[0] // 3
            operations = stacked.reshape(n_ops, 3, 3)
        else:
            fn_map = {"T": _tetrahedral_ops_py, "O": _octahedral_ops_py, "I": _icosahedral_ops_py}
            if name not in fn_map:
                raise ValueError(f"Unknown symmetry group '{name}'. Expected 'T', 'O', or 'I'.")
            ops = fn_map[name]()
            operations = np.array(ops)

        return cls(name=name, operations=operations)

    @property
    def order(self) -> int:
        return self.operations.shape[0]

    @property
    def info(self) -> dict:
        """Return symmetry group metadata."""
        return _SYMMETRY_INFO.get(self.name, {})

    def expand_coords(self, coords: NDArray) -> NDArray:
        """Apply all symmetry operations to a set of coordinates.

        Parameters
        ----------
        coords : NDArray
            Shape (N, 3) — coordinates of the asymmetric unit.

        Returns
        -------
        NDArray
            Shape (N * order, 3) — symmetry-expanded coordinates.
        """
        if _HAS_RUST:
            stacked = self.operations.reshape(-1, 3)  # (M*3, 3)
            return _rs_apply_symmetry_ops(
                np.ascontiguousarray(coords, dtype=np.float64),
                np.ascontiguousarray(stacked, dtype=np.float64),
            )
        else:
            parts = []
            for R in self.operations:
                parts.append((coords @ R.T))
            return np.vstack(parts)

    def expand_with_chain_ids(self, coords: NDArray) -> tuple[NDArray, NDArray]:
        """Expand coordinates and return chain ID array.

        Returns
        -------
        tuple
            (expanded_coords (N*M, 3), chain_ids (N*M,)) where chain_ids[i]
            is the symmetry operation index (0..M-1) for each atom.
        """
        expanded = self.expand_coords(coords)
        n_atoms = coords.shape[0]
        chain_ids = np.repeat(np.arange(self.order), n_atoms)
        return expanded, chain_ids

    def get_symmetry_axes(self) -> dict[str, NDArray]:
        """Return symmetry axes as unit vectors.

        Returns
        -------
        dict
            Keys like "C2", "C3", "C5" → NDArray of shape (K, 3) unit vectors.
        """
        axes = {}
        if self.name == "T":
            # C2 axes: along (1,0,0), (0,1,0), (0,0,1)
            axes["C2"] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            # C3 axes: along (±1,±1,±1)/√3
            c3 = []
            for sx in [1, -1]:
                for sy in [1, -1]:
                    c3.append([sx, sy, sx * sy])
            axes["C3"] = np.array(c3, dtype=float) / np.sqrt(3)
        elif self.name == "O":
            axes["C4"] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            c3 = []
            for sx in [1, -1]:
                for sy in [1, -1]:
                    c3.append([sx, sy, sx * sy])
            axes["C3"] = np.array(c3, dtype=float) / np.sqrt(3)
            c2 = []
            for pair in [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]:
                c2.append(list(pair))
            axes["C2"] = np.array(c2, dtype=float) / np.sqrt(2)
        elif self.name == "I":
            # C5 axes: 6 axes through opposite vertices of icosahedron
            phi = _PHI
            norm = np.sqrt(1 + phi**2)
            c5 = [
                [0, 1 / norm, phi / norm],
                [0, -1 / norm, phi / norm],
                [0, 1 / norm, -phi / norm],
                [1 / norm, phi / norm, 0],
                [-1 / norm, phi / norm, 0],
                [phi / norm, 0, 1 / norm],
            ]
            axes["C5"] = np.array(c5, dtype=float)
            # C3 axes through face centres — use (1,1,1) family + icosahedral variants
            c3 = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                           [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                           [0, phi, 1], [0, -phi, 1]], dtype=float)
            c3 = c3 / np.linalg.norm(c3, axis=1, keepdims=True)
            axes["C3"] = c3
            # C2 axes through edge midpoints
            axes["C2"] = np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            ], dtype=float)
            axes["C2"] = axes["C2"] / np.linalg.norm(axes["C2"], axis=1, keepdims=True)

        return axes
