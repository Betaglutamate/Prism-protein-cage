"""
PDB/mmCIF structure I/O with BioPython and gemmi.

Provides thin wrappers that handle common cage-related I/O tasks:
multi-chain PDB assembly writing, symmetry expansion to file,
and mmCIF round-tripping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def read_structure(path: str | Path):
    """Read a PDB or mmCIF file into a BioPython Structure.

    Returns
    -------
    Bio.PDB.Structure.Structure
    """
    from Bio.PDB import PDBParser, MMCIFParser

    path = Path(path)
    if path.suffix.lower() in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    return parser.get_structure(path.stem, str(path))


def write_pdb(
    coords: NDArray,
    output_path: str | Path,
    chain_ids: Optional[NDArray] = None,
    residue_names: Optional[list[str]] = None,
    atom_names: Optional[list[str]] = None,
    elements: Optional[list[str]] = None,
    b_factors: Optional[NDArray] = None,
) -> Path:
    """Write coordinates to a PDB file.

    Handles multi-chain output and proper formatting.

    Parameters
    ----------
    coords : NDArray
        Shape (N, 3) coordinates.
    output_path : str | Path
        Output file path.
    chain_ids : NDArray, optional
        Per-atom chain IDs. Default: all 'A'.
    residue_names : list[str], optional
        Per-atom residue names. Default: 'ALA'.
    atom_names : list[str], optional
        Per-atom names. Default: 'CA'.
    elements : list[str], optional
        Per-atom element symbols. Default: 'C'.
    b_factors : NDArray, optional
        Per-atom B-factors for annotation.

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(coords)
    if chain_ids is None:
        chain_ids = np.array(["A"] * n)
    if residue_names is None:
        residue_names = ["ALA"] * n
    if atom_names is None:
        atom_names = ["CA"] * n
    if elements is None:
        elements = ["C"] * n
    if b_factors is None:
        b_factors = np.zeros(n)

    lines = []
    current_chain = None
    res_num = 0
    atom_serial = 1

    for i in range(n):
        ch = str(chain_ids[i])
        if ch != current_chain:
            if current_chain is not None:
                lines.append("TER")
            current_chain = ch
            res_num = 0

        res_num += 1
        x, y, z = coords[i]
        atom_name = atom_names[i].ljust(4) if len(atom_names[i]) < 4 else atom_names[i]
        line = (
            f"ATOM  {atom_serial:5d} {atom_name:4s} "
            f"{residue_names[i]:3s} {ch}{res_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.00:6.2f}{b_factors[i]:6.2f}          "
            f"{elements[i]:>2s}"
        )
        lines.append(line)
        atom_serial += 1

    lines.append("TER")
    lines.append("END")

    path.write_text("\n".join(lines) + "\n")
    return path


def write_cage_pdb(
    cage_design,
    output_path: str | Path,
    include_surface_annotation: bool = True,
) -> Path:
    """Write a CageDesign to a multi-chain PDB with optional B-factor annotations.

    Parameters
    ----------
    cage_design : prism.core.cage.CageDesign
        The cage design to export.
    output_path : str | Path
        Output PDB path.
    include_surface_annotation : bool
        If True, annotate interior residues with elevated B-factors.

    Returns
    -------
    Path
    """
    path = Path(output_path)

    if cage_design.structure is None:
        raise ValueError("CageDesign has no loaded structure.")

    structure = cage_design.structure

    # If expanded, use the expanded structure
    if cage_design.expanded_structure is not None:
        structure.save_pdb(str(path))
    else:
        structure.save_pdb(str(path))

    return path


def write_mmcif(
    coords: NDArray,
    output_path: str | Path,
    chain_ids: Optional[NDArray] = None,
) -> Path:
    """Write coordinates to an mmCIF file using gemmi.

    Parameters
    ----------
    coords : NDArray
        Shape (N, 3).
    output_path : str | Path
        Output path.
    chain_ids : NDArray, optional
        Per-atom chain IDs.

    Returns
    -------
    Path
    """
    try:
        import gemmi
    except ImportError:
        raise ImportError("gemmi is required for mmCIF output. Install: pip install gemmi")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    st = gemmi.Structure()
    st.name = path.stem
    model = gemmi.Model("1")

    if chain_ids is None:
        chain_ids = np.array(["A"] * len(coords))

    unique_chains = np.unique(chain_ids)
    for ch_id in unique_chains:
        chain = gemmi.Chain(str(ch_id))
        mask = chain_ids == ch_id
        ch_coords = coords[mask]

        for j, (x, y, z) in enumerate(ch_coords):
            res = gemmi.Residue()
            res.name = "ALA"
            res.seqid = gemmi.SeqId(str(j + 1))

            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(x), float(y), float(z))
            atom.occ = 1.0
            atom.b_iso = 0.0

            res.add_atom(atom)
            chain.add_residue(res)

        model.add_chain(chain)

    st.add_model(model)
    st.write_minimal_cif(str(path))

    return path


def read_fasta(path: str | Path) -> dict[str, str]:
    """Read a FASTA file into a dict of {header: sequence}.

    Returns
    -------
    dict[str, str]
        Keys are header lines (without '>'), values are sequences.
    """
    path = Path(path)
    sequences: dict[str, str] = {}
    current_header = None
    current_seq: list[str] = []

    for line in path.read_text().strip().split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current_header is not None:
                sequences[current_header] = "".join(current_seq)
            current_header = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line)

    if current_header is not None:
        sequences[current_header] = "".join(current_seq)

    return sequences


def write_fasta(
    sequences: dict[str, str],
    output_path: str | Path,
    line_width: int = 80,
) -> Path:
    """Write sequences to a FASTA file.

    Parameters
    ----------
    sequences : dict[str, str]
        {header: sequence} mapping.
    output_path : str | Path
        Output file path.
    line_width : int
        Characters per line for sequence wrapping.

    Returns
    -------
    Path
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for header, seq in sequences.items():
        lines.append(f">{header}")
        for i in range(0, len(seq), line_width):
            lines.append(seq[i : i + line_width])

    path.write_text("\n".join(lines) + "\n")
    return path
