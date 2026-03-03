"""
ProteinMPNN integration — sequence design for RFdiffusion-generated backbones.

ProteinMPNN assigns amino acid sequences to designed backbone structures,
optimising for thermodynamic stability and foldability.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ProteinMPNNConfig:
    """Configuration for a ProteinMPNN installation."""

    mpnn_dir: Path
    python_executable: str = "python"

    @property
    def main_script(self) -> Path:
        return self.mpnn_dir / "protein_mpnn_run.py"

    @property
    def helper_scripts(self) -> Path:
        return self.mpnn_dir / "helper_scripts"

    def validate(self) -> None:
        if not self.main_script.exists():
            raise FileNotFoundError(
                f"ProteinMPNN script not found: {self.main_script}"
            )


@dataclass
class SequenceDesignResult:
    """Result from a ProteinMPNN sequence design run.

    Attributes
    ----------
    sequences : list[dict]
        Each dict: {"sequence": str, "score": float, "recovery": float}.
    output_fasta : Path or None
        Path to the output FASTA file.
    output_dir : Path
        Output directory.
    config_used : dict
        Parameters used.
    """

    sequences: list[dict] = field(default_factory=list)
    output_fasta: Optional[Path] = None
    output_dir: Optional[Path] = None
    config_used: dict = field(default_factory=dict)

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)

    @property
    def best_sequence(self) -> Optional[dict]:
        """Return the sequence with the best (lowest) score."""
        if not self.sequences:
            return None
        return min(self.sequences, key=lambda s: s.get("score", float("inf")))

    def summary(self) -> str:
        lines = [
            f"ProteinMPNN Sequence Design Result",
            f"  Sequences designed: {self.n_sequences}",
        ]
        if self.sequences:
            scores = [s.get("score", float("nan")) for s in self.sequences]
            lines.append(f"  Score range: {min(scores):.2f} – {max(scores):.2f}")
            best = self.best_sequence
            if best:
                lines.append(f"  Best: score={best.get('score', '?'):.2f}")
                seq = best.get("sequence", "")
                if len(seq) > 40:
                    lines.append(f"    {seq[:40]}...")
                else:
                    lines.append(f"    {seq}")
        return "\n".join(lines)


class ProteinMPNNRunner:
    """Wrapper for ProteinMPNN sequence design.

    Examples
    --------
    >>> config = ProteinMPNNConfig(mpnn_dir=Path("/path/to/ProteinMPNN"))
    >>> runner = ProteinMPNNRunner(config)
    >>> result = runner.design_sequences(
    ...     pdb_path=Path("backbone.pdb"),
    ...     num_sequences=8,
    ...     output_dir=Path("./sequences"),
    ... )
    """

    def __init__(self, config: ProteinMPNNConfig):
        self.config = config

    def design_sequences(
        self,
        pdb_path: Path,
        num_sequences: int,
        output_dir: Path,
        *,
        temperature: float = 0.1,
        chain_id: str = "A",
        fixed_positions: Optional[list[int]] = None,
        tied_positions: Optional[list[list[int]]] = None,
        sampling_temp: float = 0.1,
        batch_size: int = 1,
    ) -> SequenceDesignResult:
        """Design sequences for a backbone structure.

        Parameters
        ----------
        pdb_path : Path
            Input backbone PDB (e.g. from RFdiffusion).
        num_sequences : int
            Number of sequence samples to generate.
        output_dir : Path
            Output directory.
        temperature : float
            Sampling temperature. Lower = more conservative.
        chain_id : str
            Chain to design. Default "A".
        fixed_positions : list[int], optional
            Residue positions to keep fixed (0-indexed).
        tied_positions : list[list[int]], optional
            Groups of positions that must have the same amino acid
            (for symmetric subunits).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Parse PDB and generate input JSONs
        # ProteinMPNN expects a parsed_pdbs.jsonl and optionally
        # assigned_pdbs.jsonl + fixed_pdbs.jsonl
        parsed_dir = output_dir / "parsed"
        parsed_dir.mkdir(exist_ok=True)

        # Run the helper parse script
        parse_cmd = [
            self.config.python_executable,
            str(self.config.helper_scripts / "parse_multiple_chains.py"),
            f"--input_path={pdb_path.parent}",
            f"--output_path={parsed_dir / 'parsed_pdbs.jsonl'}",
        ]

        subprocess.run(
            parse_cmd,
            cwd=str(self.config.mpnn_dir),
            capture_output=True,
            timeout=300,
        )

        # Step 2: Run ProteinMPNN
        cmd = [
            self.config.python_executable,
            str(self.config.main_script),
            "--jsonl_path", str(parsed_dir / "parsed_pdbs.jsonl"),
            "--out_folder", str(output_dir),
            "--num_seq_per_target", str(num_sequences),
            "--sampling_temp", str(sampling_temp),
            "--batch_size", str(batch_size),
        ]

        if fixed_positions:
            # Write fixed positions JSON
            fixed_dict = {pdb_path.stem: {chain_id: fixed_positions}}
            fixed_path = output_dir / "fixed_positions.json"
            fixed_path.write_text(json.dumps(fixed_dict))
            cmd.extend(["--fixed_positions_jsonl", str(fixed_path)])

        if tied_positions:
            tied_dict = {pdb_path.stem: [
                [{"chain": chain_id, "pos": pos} for pos in group]
                for group in tied_positions
            ]}
            tied_path = output_dir / "tied_positions.json"
            tied_path.write_text(json.dumps(tied_dict))
            cmd.extend(["--tied_positions_jsonl", str(tied_path)])

        log_path = output_dir / "proteinmpnn.log"
        with open(log_path, "w") as log_file:
            process = subprocess.run(
                cmd,
                cwd=str(self.config.mpnn_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )

        if process.returncode != 0:
            log_content = log_path.read_text()[-2000:]
            raise RuntimeError(
                f"ProteinMPNN failed (exit code {process.returncode}).\n"
                f"Log tail:\n{log_content}"
            )

        # Parse output FASTA
        sequences = self._parse_output(output_dir)

        return SequenceDesignResult(
            sequences=sequences,
            output_dir=output_dir,
            config_used={
                "pdb_path": str(pdb_path),
                "num_sequences": num_sequences,
                "temperature": temperature,
                "sampling_temp": sampling_temp,
            },
        )

    @staticmethod
    def _parse_output(output_dir: Path) -> list[dict]:
        """Parse ProteinMPNN output FASTA files."""
        sequences = []
        seqs_dir = output_dir / "seqs"
        if not seqs_dir.exists():
            # Try flat structure
            fasta_files = list(output_dir.glob("*.fa")) + list(output_dir.glob("*.fasta"))
        else:
            fasta_files = list(seqs_dir.glob("*.fa")) + list(seqs_dir.glob("*.fasta"))

        for fasta_path in fasta_files:
            current_header = ""
            current_seq = ""
            with open(fasta_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_seq:
                            score = _parse_score_from_header(current_header)
                            sequences.append({
                                "sequence": current_seq,
                                "score": score,
                                "header": current_header,
                            })
                        current_header = line[1:]
                        current_seq = ""
                    elif line:
                        current_seq += line
                # Don't forget last entry
                if current_seq:
                    score = _parse_score_from_header(current_header)
                    sequences.append({
                        "sequence": current_seq,
                        "score": score,
                        "header": current_header,
                    })

        return sequences

    @staticmethod
    def check_installation(mpnn_dir: Path) -> dict:
        """Check if ProteinMPNN is properly installed."""
        result = {
            "installed": False,
            "main_script": False,
            "helper_scripts": False,
            "issues": [],
        }

        script = mpnn_dir / "protein_mpnn_run.py"
        if script.exists():
            result["main_script"] = True
        else:
            result["issues"].append(f"protein_mpnn_run.py not found")

        helpers = mpnn_dir / "helper_scripts"
        if helpers.exists():
            result["helper_scripts"] = True
        else:
            result["issues"].append(f"helper_scripts/ not found")

        result["installed"] = result["main_script"] and result["helper_scripts"]
        return result


def _parse_score_from_header(header: str) -> float:
    """Extract score from a ProteinMPNN FASTA header.

    Headers typically look like: ">design_0, score=1.234, ..."
    """
    try:
        for part in header.split(","):
            part = part.strip()
            if part.startswith("score="):
                return float(part.split("=")[1])
    except (ValueError, IndexError):
        pass
    return float("nan")
