"""
BindCraft integration — design docking interfaces for cage-cage assembly.

Wraps the BindCraft CLI to design protein binders for exterior cage surfaces,
enabling programmed self-assembly of cages into lattice architectures.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BindCraftConfig:
    """Configuration for a BindCraft installation.

    Parameters
    ----------
    bindcraft_dir : Path
        Root directory of the BindCraft installation.
    python_executable : str
        Python executable (may differ from system python if BindCraft
        has its own conda environment).
    """

    bindcraft_dir: Path
    python_executable: str = "python"

    @property
    def main_script(self) -> Path:
        return self.bindcraft_dir / "bindcraft.py"

    def validate(self) -> None:
        if not self.main_script.exists():
            raise FileNotFoundError(
                f"BindCraft script not found: {self.main_script}"
            )


@dataclass
class InterfaceDesignResult:
    """Result from a BindCraft interface design run.

    Attributes
    ----------
    output_pdbs : list[Path]
        Designed complex PDB files (binder + target).
    metrics : list[dict]
        Per-design metrics (pAE, pLDDT, iPTM, binding energy, etc.).
    sequences : list[str]
        Designed binder sequences.
    config_used : dict
        Parameters used for this run.
    log_path : Path or None
        Stdout/stderr log.
    """

    output_pdbs: list[Path] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)
    config_used: dict = field(default_factory=dict)
    log_path: Optional[Path] = None

    @property
    def n_designs(self) -> int:
        return len(self.output_pdbs)

    def best_by_metric(self, metric: str = "pae_interaction", ascending: bool = True) -> dict:
        """Return the best design by a given metric.

        Parameters
        ----------
        metric : str
            Metric name (e.g. "pae_interaction", "plddt", "iptm").
        ascending : bool
            If True, lower is better (e.g., pAE). If False, higher is better (e.g., pLDDT).
        """
        if not self.metrics:
            raise ValueError("No metrics available.")

        sorted_metrics = sorted(
            enumerate(self.metrics),
            key=lambda x: x[1].get(metric, float("inf")),
            reverse=not ascending,
        )
        idx, best = sorted_metrics[0]
        return {
            "index": idx,
            "pdb": self.output_pdbs[idx] if idx < len(self.output_pdbs) else None,
            "sequence": self.sequences[idx] if idx < len(self.sequences) else None,
            **best,
        }

    def summary(self) -> str:
        lines = [
            f"BindCraft Interface Design Result",
            f"  Designs generated: {self.n_designs}",
        ]
        if self.metrics:
            paes = [m.get("pae_interaction", float("nan")) for m in self.metrics]
            plddts = [m.get("plddt", float("nan")) for m in self.metrics]
            lines.append(f"  pAE interaction: {min(paes):.1f} – {max(paes):.1f}")
            lines.append(f"  pLDDT: {min(plddts):.1f} – {max(plddts):.1f}")
        return "\n".join(lines)


# ── Default filter settings ─────────────────────────────────────────

DEFAULT_FILTERS = {
    "pae_interaction_cutoff": 10.0,
    "plddt_cutoff": 70.0,
    "iptm_cutoff": 0.6,
    "clash_cutoff": 0.5,
}


class BindCraftRunner:
    """Wrapper for invoking BindCraft to design docking interfaces.

    Examples
    --------
    >>> config = BindCraftConfig(bindcraft_dir=Path("/path/to/BindCraft"))
    >>> runner = BindCraftRunner(config)
    >>> result = runner.design_docking_interface(
    ...     cage_pdb=Path("cage_subunit.pdb"),
    ...     interface_residues=[45, 48, 52, 55, 59],
    ...     binder_length=(50, 80),
    ...     num_designs=50,
    ...     output_dir=Path("./interface_designs"),
    ... )
    """

    def __init__(self, config: BindCraftConfig):
        self.config = config

    def design_docking_interface(
        self,
        cage_pdb: Path,
        interface_residues: list[int],
        binder_length: tuple[int, int],
        num_designs: int,
        output_dir: Path,
        *,
        partner_cage_pdb: Optional[Path] = None,
        target_chain: str = "A",
        filter_settings: Optional[dict] = None,
        extra_args: Optional[dict] = None,
    ) -> InterfaceDesignResult:
        """Design a docking interface for cage-cage assembly.

        Parameters
        ----------
        cage_pdb : Path
            PDB file of the cage surface to design a binder for.
        interface_residues : list[int]
            Residue numbers on the cage exterior that define the binding patch.
        binder_length : tuple[int, int]
            (min, max) binder length in residues.
        num_designs : int
            Number of designs to attempt.
        output_dir : Path
            Output directory for designs.
        partner_cage_pdb : Path, optional
            If designing a heteromeric interface, the partner cage PDB.
            If None, designs a self-complementary (homomeric) interface.
        target_chain : str
            Chain ID of the target in the PDB. Default "A".
        filter_settings : dict, optional
            Override default filter thresholds.
        extra_args : dict, optional
            Additional CLI arguments.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine target PDB
        target = partner_cage_pdb if partner_cage_pdb else cage_pdb

        # Build filter settings
        filters = {**DEFAULT_FILTERS}
        if filter_settings:
            filters.update(filter_settings)

        filter_path = output_dir / "filters.json"
        filter_path.write_text(json.dumps(filters, indent=2))

        # Build command
        hotspot_str = ",".join(str(r) for r in interface_residues)
        min_len, max_len = binder_length

        cmd = [
            self.config.python_executable,
            str(self.config.main_script),
            "--target_pdb", str(target),
            "--target_chain", target_chain,
            "--target_hotspot_residues", hotspot_str,
            "--binder_length", f"{min_len}-{max_len}",
            "--num_designs", str(num_designs),
            "--output_dir", str(output_dir),
            "--filters", str(filter_path),
        ]

        if extra_args:
            for key, val in extra_args.items():
                cmd.extend([f"--{key}", str(val)])

        # Run BindCraft
        log_path = output_dir / "bindcraft.log"
        with open(log_path, "w") as log_file:
            process = subprocess.run(
                cmd,
                cwd=str(self.config.bindcraft_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=3600 * 8,  # 8 hour timeout for large runs
            )

        if process.returncode != 0:
            log_content = log_path.read_text()[-2000:]
            raise RuntimeError(
                f"BindCraft failed (exit code {process.returncode}).\n"
                f"Log tail:\n{log_content}"
            )

        # Parse outputs
        output_pdbs = sorted(output_dir.glob("*.pdb"))
        metrics = self._parse_metrics(output_dir)
        sequences = self._parse_sequences(output_dir)

        return InterfaceDesignResult(
            output_pdbs=output_pdbs,
            metrics=metrics,
            sequences=sequences,
            config_used={
                "cage_pdb": str(cage_pdb),
                "interface_residues": interface_residues,
                "binder_length": list(binder_length),
                "num_designs": num_designs,
                "filters": filters,
            },
            log_path=log_path,
        )

    # ── Output parsing ──────────────────────────────────────────────

    @staticmethod
    def _parse_metrics(output_dir: Path) -> list[dict]:
        """Parse BindCraft metrics from CSV output."""
        metrics = []
        for csv_path in output_dir.glob("*.csv"):
            try:
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        parsed = {}
                        for k, v in row.items():
                            try:
                                parsed[k] = float(v)
                            except (ValueError, TypeError):
                                parsed[k] = v
                        metrics.append(parsed)
            except Exception:
                continue
        return metrics

    @staticmethod
    def _parse_sequences(output_dir: Path) -> list[str]:
        """Parse designed sequences from FASTA files."""
        sequences = []
        for fasta_path in output_dir.glob("*.fasta"):
            try:
                with open(fasta_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith(">"):
                            sequences.append(line)
            except Exception:
                continue
        # Also try .fa extension
        for fasta_path in output_dir.glob("*.fa"):
            try:
                with open(fasta_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith(">"):
                            sequences.append(line)
            except Exception:
                continue
        return sequences

    @staticmethod
    def check_installation(bindcraft_dir: Path) -> dict:
        """Check if BindCraft is properly installed."""
        result = {
            "installed": False,
            "main_script": False,
            "issues": [],
        }

        script = bindcraft_dir / "bindcraft.py"
        if script.exists():
            result["main_script"] = True
        else:
            result["issues"].append(f"bindcraft.py not found at {script}")

        result["installed"] = result["main_script"]
        return result
