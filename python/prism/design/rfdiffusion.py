"""
RFdiffusion integration — invoke RFdiffusion for cage backbone generation.

Wraps the RFdiffusion CLI (Hydra-based) to generate protein cage backbones
in symmetry mode. Builds contigmaps, manages output parsing, and returns
structured results.

RFdiffusion supports tetrahedral, octahedral, and icosahedral symmetry modes
natively, which is directly applicable to PRISM cage design.
"""

from __future__ import annotations

import json
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from prism.core.cage import CageSpec


@dataclass
class RFdiffusionConfig:
    """Configuration for an RFdiffusion installation.

    Parameters
    ----------
    rfdiffusion_dir : Path
        Root directory of the RFdiffusion installation (contains run_inference.py).
    model_dir : Path
        Path to model weights directory.
    python_executable : str
        Python executable to use (e.g. path to a conda env python).
    """

    rfdiffusion_dir: Path
    model_dir: Path
    python_executable: str = "python"

    @property
    def inference_script(self) -> Path:
        return self.rfdiffusion_dir / "run_inference.py"

    def validate(self) -> None:
        """Check that paths exist."""
        if not self.inference_script.exists():
            raise FileNotFoundError(
                f"RFdiffusion inference script not found: {self.inference_script}"
            )
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")


@dataclass
class CageDesignResult:
    """Result from an RFdiffusion cage backbone generation run.

    Attributes
    ----------
    output_pdbs : list[Path]
        Paths to generated backbone PDB files.
    config_used : dict
        The Hydra overrides passed to RFdiffusion.
    cage_spec : CageSpec
        The cage specification that was used.
    log_path : Path or None
        Path to the stdout/stderr log.
    """

    output_pdbs: list[Path] = field(default_factory=list)
    config_used: dict = field(default_factory=dict)
    cage_spec: Optional[CageSpec] = None
    log_path: Optional[Path] = None

    @property
    def n_designs(self) -> int:
        return len(self.output_pdbs)

    def summary(self) -> str:
        lines = [
            f"RFdiffusion Design Result",
            f"  Designs generated: {self.n_designs}",
        ]
        if self.cage_spec:
            lines.append(f"  Symmetry: {self.cage_spec.symmetry_group}")
            lines.append(f"  Subunit length: {self.cage_spec.subunit_length_range}")
        for p in self.output_pdbs[:5]:
            lines.append(f"    → {p.name}")
        if self.n_designs > 5:
            lines.append(f"    ... and {self.n_designs - 5} more")
        return "\n".join(lines)


# ── Symmetry type mapping ───────────────────────────────────────────

_SYMMETRY_MAP = {
    "T": "tetrahedral",
    "O": "octahedral",
    "I": "icosahedral",
    "tetrahedral": "tetrahedral",
    "octahedral": "octahedral",
    "icosahedral": "icosahedral",
}


class RFdiffusionRunner:
    """Wrapper for invoking RFdiffusion to design cage backbones.

    Examples
    --------
    >>> config = RFdiffusionConfig(
    ...     rfdiffusion_dir=Path("/path/to/RFdiffusion"),
    ...     model_dir=Path("/path/to/models"),
    ... )
    >>> runner = RFdiffusionRunner(config)
    >>> result = runner.design_cage_subunit(
    ...     symmetry_type="I",
    ...     subunit_length=(80, 120),
    ...     num_designs=10,
    ...     output_dir=Path("./designs"),
    ... )
    """

    def __init__(self, config: RFdiffusionConfig):
        self.config = config

    def design_cage_subunit(
        self,
        symmetry_type: str,
        subunit_length: tuple[int, int],
        num_designs: int,
        output_dir: Path,
        *,
        input_pdb: Optional[Path] = None,
        potentials: Optional[dict] = None,
        guide_scale: float = 1.0,
        extra_overrides: Optional[dict] = None,
    ) -> CageDesignResult:
        """Design cage subunit backbones using RFdiffusion symmetry mode.

        Parameters
        ----------
        symmetry_type : str
            Symmetry group: "T", "O", "I" (or full name).
        subunit_length : tuple[int, int]
            (min_residues, max_residues) for the designed subunit.
        num_designs : int
            Number of backbone designs to generate.
        output_dir : Path
            Directory for output PDB files.
        input_pdb : Path, optional
            Template PDB for scaffold-guided design.
        potentials : dict, optional
            Guiding potentials (e.g. {"monomer_ROG": {"weight": 1.0}}).
        guide_scale : float
            Scale factor for guiding potentials.
        extra_overrides : dict, optional
            Additional Hydra overrides to pass to RFdiffusion.

        Returns
        -------
        CageDesignResult
            Contains paths to generated PDB files and design metadata.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sym = _SYMMETRY_MAP.get(symmetry_type)
        if sym is None:
            raise ValueError(
                f"Unknown symmetry type '{symmetry_type}'. "
                f"Expected one of: {list(_SYMMETRY_MAP.keys())}"
            )

        # Build contigmap
        min_res, max_res = subunit_length
        contig = f"[{min_res}-{max_res}]"

        # Build Hydra overrides
        overrides = {
            "inference.output_prefix": str(output_dir / "design"),
            "inference.model_directory_path": str(self.config.model_dir),
            "inference.num_designs": str(num_designs),
            "contigmap.contigs": contig,
            f"symmetry.symmetry_type": sym,
        }

        if input_pdb is not None:
            overrides["inference.input_pdb"] = str(input_pdb)

        if potentials:
            for pot_name, pot_config in potentials.items():
                for key, val in pot_config.items():
                    overrides[f"potentials.guiding_potentials.{pot_name}.{key}"] = str(val)
            overrides["potentials.guide_scale"] = str(guide_scale)

        if extra_overrides:
            overrides.update({k: str(v) for k, v in extra_overrides.items()})

        # Build command
        cmd = [
            self.config.python_executable,
            str(self.config.inference_script),
        ]
        for key, val in overrides.items():
            cmd.append(f"{key}={val}")

        # Run RFdiffusion
        log_path = output_dir / "rfdiffusion.log"
        with open(log_path, "w") as log_file:
            process = subprocess.run(
                cmd,
                cwd=str(self.config.rfdiffusion_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=3600 * 4,  # 4 hour timeout
            )

        if process.returncode != 0:
            log_content = log_path.read_text()[-2000:]  # last 2000 chars
            raise RuntimeError(
                f"RFdiffusion failed (exit code {process.returncode}).\n"
                f"Log tail:\n{log_content}"
            )

        # Collect output PDBs
        output_pdbs = sorted(output_dir.glob("design_*.pdb"))

        return CageDesignResult(
            output_pdbs=output_pdbs,
            config_used=overrides,
            log_path=log_path,
        )

    def design_binder(
        self,
        target_pdb: Path,
        hotspot_residues: list[str],
        binder_length: tuple[int, int],
        num_designs: int,
        output_dir: Path,
        *,
        extra_overrides: Optional[dict] = None,
    ) -> CageDesignResult:
        """Design a binder to a target surface (for docking interface fragments).

        Parameters
        ----------
        target_pdb : Path
            PDB file of the target protein surface.
        hotspot_residues : list[str]
            Hotspot residues in format ["A30", "A33", "A34"] (chain+resnum).
        binder_length : tuple[int, int]
            (min, max) binder length in residues.
        num_designs : int
            Number of designs to generate.
        output_dir : Path
            Output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        min_res, max_res = binder_length
        hotspot_str = "[" + ",".join(hotspot_residues) + "]"

        overrides = {
            "inference.output_prefix": str(output_dir / "binder"),
            "inference.model_directory_path": str(self.config.model_dir),
            "inference.input_pdb": str(target_pdb),
            "inference.num_designs": str(num_designs),
            "contigmap.contigs": f"[{min_res}-{max_res}/0 A1-999]",
            "ppi.hotspot_res": hotspot_str,
        }

        if extra_overrides:
            overrides.update({k: str(v) for k, v in extra_overrides.items()})

        cmd = [self.config.python_executable, str(self.config.inference_script)]
        for key, val in overrides.items():
            cmd.append(f"{key}={val}")

        log_path = output_dir / "rfdiffusion_binder.log"
        with open(log_path, "w") as log_file:
            process = subprocess.run(
                cmd,
                cwd=str(self.config.rfdiffusion_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=3600 * 4,
            )

        if process.returncode != 0:
            log_content = log_path.read_text()[-2000:]
            raise RuntimeError(
                f"RFdiffusion binder design failed (exit code {process.returncode}).\n"
                f"Log tail:\n{log_content}"
            )

        output_pdbs = sorted(output_dir.glob("binder_*.pdb"))
        return CageDesignResult(
            output_pdbs=output_pdbs,
            config_used=overrides,
            log_path=log_path,
        )

    @staticmethod
    def check_installation(rfdiffusion_dir: Path) -> dict:
        """Check if RFdiffusion is properly installed.

        Returns
        -------
        dict
            {"installed": bool, "inference_script": bool,
             "models_found": list[str], "issues": list[str]}
        """
        result = {
            "installed": False,
            "inference_script": False,
            "models_found": [],
            "issues": [],
        }

        inference = rfdiffusion_dir / "run_inference.py"
        if inference.exists():
            result["inference_script"] = True
        else:
            result["issues"].append(f"run_inference.py not found at {inference}")

        models_dir = rfdiffusion_dir / "models"
        if models_dir.exists():
            result["models_found"] = [p.name for p in models_dir.glob("*.pt")]
        else:
            result["issues"].append(f"models/ directory not found at {models_dir}")

        result["installed"] = result["inference_script"] and len(result["models_found"]) > 0
        return result
