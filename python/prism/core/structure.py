"""
Thin structural biology wrapper around BioPython / gemmi for PDB I/O.

Provides a `ProteinStructure` class with convenience methods for
extracting coordinates, residue information, and chain selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class ProteinStructure:
    """Lightweight wrapper around a protein structure (PDB/mmCIF).

    Wraps BioPython's Structure object with convenience methods for
    coordinate extraction, chain selection, and residue queries that
    are used throughout the PRISM pipeline.

    Parameters
    ----------
    path : Path or str, optional
        Path to a PDB or mmCIF file. If provided, the structure is loaded
        immediately.
    """

    def __init__(self, path: Optional[str | Path] = None):
        self._structure = None
        self._path: Optional[Path] = None
        self._coords_cache: Optional[NDArray] = None

        if path is not None:
            self.load(path)

    def load(self, path: str | Path) -> ProteinStructure:
        """Load a structure from PDB or mmCIF file."""
        from Bio.PDB import PDBParser, MMCIFParser

        path = Path(path)
        self._path = path
        self._coords_cache = None

        if path.suffix in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        self._structure = parser.get_structure(path.stem, str(path))
        return self

    @classmethod
    def from_biopython(cls, structure) -> ProteinStructure:
        """Create from an existing BioPython Structure object."""
        obj = cls()
        obj._structure = structure
        return obj

    @property
    def structure(self):
        """Underlying BioPython Structure object."""
        if self._structure is None:
            raise RuntimeError("No structure loaded. Call .load() first.")
        return self._structure

    @property
    def path(self) -> Optional[Path]:
        return self._path

    # ── Coordinate extraction ───────────────────────────────────────

    def get_all_coords(self) -> NDArray:
        """Return all protein atom coordinates as (N, 3) numpy array.

        Excludes hetero atoms and water.
        """
        if self._coords_cache is not None:
            return self._coords_cache
        coords = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] != " ":
                        continue
                    for atom in residue:
                        coords.append(atom.get_vector().get_array())
        arr = np.array(coords, dtype=np.float64) if coords else np.zeros((0, 3))
        self._coords_cache = arr
        return arr

    def get_ca_coords(self, chain_id: Optional[str] = None) -> NDArray:
        """Return Cα coordinates as (R, 3) array.

        Parameters
        ----------
        chain_id : str, optional
            If given, only return Cα from the specified chain.
        """
        cas = []
        for model in self.structure:
            for chain in model:
                if chain_id is not None and chain.get_id() != chain_id:
                    continue
                for residue in chain:
                    if "CA" in residue:
                        cas.append(residue["CA"].get_vector().get_array())
        return np.array(cas, dtype=np.float64)

    def get_cb_coords(self, chain_id: Optional[str] = None) -> NDArray:
        """Return Cβ coordinates (Cα for glycine) as (R, 3) array."""
        cbs = []
        for model in self.structure:
            for chain in model:
                if chain_id is not None and chain.get_id() != chain_id:
                    continue
                for residue in chain:
                    if "CB" in residue:
                        cbs.append(residue["CB"].get_vector().get_array())
                    elif "CA" in residue:
                        # Glycine: use Cα as proxy
                        cbs.append(residue["CA"].get_vector().get_array())
        return np.array(cbs, dtype=np.float64)

    def get_elements(self) -> NDArray:
        """Return atomic numbers for all atoms as (N,) uint8 array."""
        element_map = {
            "C": 6, "N": 7, "O": 8, "S": 16, "H": 1, "P": 15, "FE": 26,
            "ZN": 30, "MG": 12, "CA": 20, "MN": 25, "CO": 27, "NI": 28,
            "CU": 29, "SE": 34,
        }
        elements = []
        for atom in self.structure.get_atoms():
            elem = atom.element.strip().upper()
            elements.append(element_map.get(elem, 6))  # default to C
        return np.array(elements, dtype=np.uint8)

    # ── Chain / residue queries ─────────────────────────────────────

    def get_chain_ids(self) -> list[str]:
        """Return list of chain IDs."""
        chains = set()
        for model in self.structure:
            for chain in model:
                chains.add(chain.get_id())
        return sorted(chains)

    def get_per_atom_chain_ids(self) -> NDArray:
        """Return per-atom chain ID array matching get_all_coords() order."""
        ids = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] != " ":
                        continue
                    for atom in residue:
                        ids.append(chain.get_id())
        return np.array(ids)

    def get_residues(self, chain_id: Optional[str] = None) -> list:
        """Return list of BioPython Residue objects."""
        residues = []
        for model in self.structure:
            for chain in model:
                if chain_id is not None and chain.get_id() != chain_id:
                    continue
                for residue in chain:
                    # Skip hetero residues and water
                    if residue.get_id()[0] == " ":
                        residues.append(residue)
        return residues

    def get_residue_names(self, chain_id: Optional[str] = None) -> list[str]:
        """Return 3-letter residue names."""
        return [r.get_resname() for r in self.get_residues(chain_id)]

    def n_residues(self, chain_id: Optional[str] = None) -> int:
        return len(self.get_residues(chain_id))

    def n_atoms(self) -> int:
        return len(list(self.structure.get_atoms()))

    def n_chains(self) -> int:
        return len(self.get_chain_ids())

    # ── Coordinate manipulation ──────────────────────────────────────

    def center_at_origin(self) -> NDArray:
        """Return coordinates centred at the centroid."""
        coords = self.get_all_coords()
        centroid = coords.mean(axis=0)
        return coords - centroid

    def get_centroid(self) -> NDArray:
        """Return the centroid of all atoms."""
        return self.get_all_coords().mean(axis=0)

    # ── I/O ──────────────────────────────────────────────────────────

    def save_pdb(self, path: str | Path) -> None:
        """Write structure to PDB file."""
        from Bio.PDB import PDBIO

        io = PDBIO()
        io.set_structure(self.structure)
        io.save(str(path))

    def __repr__(self) -> str:
        if self._structure is None:
            return "ProteinStructure(empty)"
        n = self.n_atoms()
        chains = self.get_chain_ids()
        src = f" from={self._path.name}" if self._path else ""
        return f"ProteinStructure(atoms={n}, chains={chains}{src})"
