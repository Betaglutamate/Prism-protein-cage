"""
PRISM — Programmable protein cages as a universal nanocrystal manufacturing platform.

A computational design toolkit for engineering de novo protein cages
that template deterministic nanocrystal formation.

Modules
-------
core        Symmetry groups, structure wrappers, cage specs, surface chemistry
design      RFdiffusion, BindCraft, ProteinMPNN integrations
assembly    Docking interface identification, lattice construction
analysis    Cavity analysis, clash detection, quality metrics
viz         Interactive 3D visualization with py3Dmol
io          PDB/mmCIF/FASTA I/O, project management
cli         Command-line interface
"""

__version__ = "0.1.0"

from prism.core.symmetry import SymmetryGroup
from prism.core.cage import CageDesign, CageSpec, CavitySpec, InterfaceSpec
from prism.core.structure import ProteinStructure
from prism.core.residue_surface import SurfaceChemSpec

__all__ = [
    "SymmetryGroup",
    "CageDesign",
    "CageSpec",
    "CavitySpec",
    "InterfaceSpec",
    "ProteinStructure",
    "SurfaceChemSpec",
]
