"""
Lattice builder — tile protein cages into 3D crystal lattices.

Takes a cage assembly and lattice parameters, generates the full
multi-cage lattice by periodic translation, with support for common
lattice types (SC, BCC, FCC, hexagonal) and custom lattice vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

# Try Rust core
try:
    from prism._rust_core import build_lattice as _rs_build_lattice
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


# ── Standard lattice vectors ────────────────────────────────────────

def _sc_vectors(a: float) -> NDArray:
    """Simple cubic lattice vectors."""
    return np.array([
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a],
    ], dtype=np.float64)


def _bcc_vectors(a: float) -> NDArray:
    """Body-centred cubic lattice vectors."""
    h = a / 2
    return np.array([
        [h, h, -h],
        [h, -h, h],
        [-h, h, h],
    ], dtype=np.float64)


def _fcc_vectors(a: float) -> NDArray:
    """Face-centred cubic lattice vectors."""
    h = a / 2
    return np.array([
        [0, h, h],
        [h, 0, h],
        [h, h, 0],
    ], dtype=np.float64)


def _hex_vectors(a: float, c: Optional[float] = None) -> NDArray:
    """Hexagonal lattice vectors."""
    if c is None:
        c = a * np.sqrt(8 / 3)  # ideal c/a ratio
    return np.array([
        [a, 0, 0],
        [a / 2, a * np.sqrt(3) / 2, 0],
        [0, 0, c],
    ], dtype=np.float64)


LATTICE_GENERATORS = {
    "sc": _sc_vectors,
    "bcc": _bcc_vectors,
    "fcc": _fcc_vectors,
    "hex": _hex_vectors,
}


@dataclass
class LatticeSpec:
    """Specification for a cage lattice.

    Attributes
    ----------
    lattice_type : str
        "sc", "bcc", "fcc", "hex", or "custom".
    lattice_constant : float
        Lattice constant 'a' in Angstroms (edge length of conventional cell).
    lattice_vectors : NDArray
        Shape (3, 3) — lattice vectors as rows.
    repeats : tuple[int, int, int]
        Number of unit cells along each lattice vector.
    """

    lattice_type: str = "sc"
    lattice_constant: float = 200.0  # Å — typical for protein cage lattice
    lattice_vectors: Optional[NDArray] = field(default=None)
    repeats: tuple[int, int, int] = (3, 3, 3)

    def __post_init__(self):
        if self.lattice_vectors is None:
            lt = self.lattice_type.lower()
            if lt in LATTICE_GENERATORS:
                self.lattice_vectors = LATTICE_GENERATORS[lt](self.lattice_constant)
            else:
                self.lattice_vectors = np.eye(3) * self.lattice_constant

    @classmethod
    def from_type(
        cls,
        lattice_type: str,
        lattice_constant: float,
        repeats: tuple[int, int, int] = (3, 3, 3),
    ) -> LatticeSpec:
        """Create a LatticeSpec from a standard lattice type.

        Parameters
        ----------
        lattice_type : str
            "sc", "bcc", "fcc", or "hex".
        lattice_constant : float
            Lattice constant in Angstroms.
        repeats : tuple
            Unit cell repetitions (na, nb, nc).
        """
        if lattice_type not in LATTICE_GENERATORS:
            raise ValueError(
                f"Unknown lattice type '{lattice_type}'. "
                f"Known: {list(LATTICE_GENERATORS.keys())}"
            )
        vectors = LATTICE_GENERATORS[lattice_type](lattice_constant)
        return cls(
            lattice_type=lattice_type,
            lattice_constant=lattice_constant,
            lattice_vectors=vectors,
            repeats=repeats,
        )

    @classmethod
    def custom(
        cls,
        lattice_vectors: NDArray,
        repeats: tuple[int, int, int] = (3, 3, 3),
    ) -> LatticeSpec:
        """Create a LatticeSpec with custom lattice vectors."""
        lv = np.asarray(lattice_vectors, dtype=np.float64)
        if lv.shape != (3, 3):
            raise ValueError("lattice_vectors must have shape (3, 3)")
        return cls(
            lattice_type="custom",
            lattice_constant=np.linalg.norm(lv[0]),
            lattice_vectors=lv,
            repeats=repeats,
        )

    @property
    def n_cells(self) -> int:
        return self.repeats[0] * self.repeats[1] * self.repeats[2]


class LatticeBuilder:
    """Build a 3D lattice of protein cages.

    Examples
    --------
    >>> from prism.assembly.lattice import LatticeBuilder, LatticeSpec
    >>> spec = LatticeSpec.from_type("sc", lattice_constant=200.0, repeats=(3, 3, 3))
    >>> builder = LatticeBuilder(spec)
    >>> lattice_coords = builder.build(cage_coords)  # (N_atoms * 27, 3)
    """

    def __init__(self, spec: LatticeSpec):
        self.spec = spec

    def build(self, cage_coords: NDArray) -> NDArray:
        """Tile cage coordinates into a full lattice.

        Parameters
        ----------
        cage_coords : NDArray
            Shape (N, 3) — coordinates of a single cage assembly.

        Returns
        -------
        NDArray
            Shape (N * n_cells, 3) — full lattice coordinates.
        """
        cage = np.ascontiguousarray(cage_coords, dtype=np.float64)
        lv = np.ascontiguousarray(self.spec.lattice_vectors, dtype=np.float64)
        repeats = list(self.spec.repeats)

        if _HAS_RUST:
            return _rs_build_lattice(cage, lv, repeats)
        else:
            return self._build_python(cage, lv, repeats)

    def build_with_ids(self, cage_coords: NDArray) -> tuple[NDArray, NDArray]:
        """Build lattice and return cage ID for each atom.

        Returns
        -------
        tuple
            (lattice_coords, cage_ids) where cage_ids[i] is the index
            of the unit cell that atom i belongs to.
        """
        lattice = self.build(cage_coords)
        n_atoms = cage_coords.shape[0]
        cage_ids = np.repeat(np.arange(self.spec.n_cells), n_atoms)
        return lattice, cage_ids

    def get_cage_centers(self) -> NDArray:
        """Return the centre positions of all cages in the lattice.

        Returns
        -------
        NDArray
            Shape (n_cells, 3) — centre of each cage.
        """
        lv = self.spec.lattice_vectors
        na, nb, nc = self.spec.repeats
        centers = []
        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    t = ia * lv[0] + ib * lv[1] + ic * lv[2]
                    centers.append(t)
        return np.array(centers, dtype=np.float64)

    @staticmethod
    def _build_python(cage: NDArray, lv: NDArray, repeats: list[int]) -> NDArray:
        """Pure Python fallback for lattice construction."""
        na, nb, nc = repeats
        n_atoms = cage.shape[0]
        total = n_atoms * na * nb * nc
        result = np.empty((total, 3), dtype=np.float64)

        idx = 0
        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    translation = ia * lv[0] + ib * lv[1] + ic * lv[2]
                    result[idx:idx + n_atoms] = cage + translation
                    idx += n_atoms

        return result

    def estimate_lattice_size(self) -> dict:
        """Estimate physical dimensions of the lattice.

        Returns
        -------
        dict
            dimensions_angstrom, dimensions_nm, n_cells, n_atoms_estimate
        """
        lv = self.spec.lattice_vectors
        na, nb, nc = self.spec.repeats
        dims = np.array([
            na * np.linalg.norm(lv[0]),
            nb * np.linalg.norm(lv[1]),
            nc * np.linalg.norm(lv[2]),
        ])
        return {
            "dimensions_angstrom": dims.tolist(),
            "dimensions_nm": (dims / 10.0).tolist(),
            "n_cells": self.spec.n_cells,
        }
