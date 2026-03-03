"""
PRISM project serialisation and management.

A ``PRISMProject`` is a directory on disk that holds:
  - prism_project.json     (metadata, spec, parameters)
  - structures/            (PDB files: subunit, expanded cage, lattice)
  - designs/               (RFdiffusion / BindCraft / MPNN outputs)
  - analysis/              (reports, metrics JSON)
  - notebooks/             (user notebooks)

This module handles creating, saving, loading, and snapshotting
design projects.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


PRISM_PROJECT_FILE = "prism_project.json"
PROJECT_VERSION = "0.1.0"


@dataclass
class DesignSnapshot:
    """A timestamped design snapshot.

    Attributes
    ----------
    timestamp : str
        ISO format timestamp.
    description : str
        Human-readable description.
    spec : dict
        Serialised CageSpec at time of snapshot.
    metrics : dict
        Quality metrics at time of snapshot.
    files : list[str]
        Relative paths to associated files.
    """

    timestamp: str
    description: str
    spec: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    files: list[str] = field(default_factory=list)


class PRISMProject:
    """Manages a PRISM protein cage design project on disk.

    Usage::

        project = PRISMProject.create("my_cage_project")
        project.set_spec(cage_spec)
        project.save()

        # Later...
        project = PRISMProject.load("my_cage_project")
    """

    def __init__(self, project_dir: str | Path):
        self.root = Path(project_dir).resolve()
        self.metadata: dict[str, Any] = {
            "version": PROJECT_VERSION,
            "created": datetime.now().isoformat(),
            "name": self.root.name,
        }
        self.spec: Optional[dict] = None
        self.parameters: dict[str, Any] = {}
        self.snapshots: list[DesignSnapshot] = []
        self.notes: list[str] = []

    # ── Directory layout ──────────────────────────────────────────────

    @property
    def structures_dir(self) -> Path:
        return self.root / "structures"

    @property
    def designs_dir(self) -> Path:
        return self.root / "designs"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    @property
    def notebooks_dir(self) -> Path:
        return self.root / "notebooks"

    def _ensure_dirs(self):
        for d in [self.structures_dir, self.designs_dir,
                   self.analysis_dir, self.notebooks_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Lifecycle ─────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        project_dir: str | Path,
        name: Optional[str] = None,
    ) -> "PRISMProject":
        """Create a new PRISM project directory.

        Parameters
        ----------
        project_dir : str | Path
            Path for the new project directory.
        name : str, optional
            Human-readable project name (default: directory name).

        Returns
        -------
        PRISMProject
        """
        proj = cls(project_dir)
        if name:
            proj.metadata["name"] = name
        proj._ensure_dirs()
        proj.save()
        return proj

    @classmethod
    def load(cls, project_dir: str | Path) -> "PRISMProject":
        """Load an existing PRISM project.

        Parameters
        ----------
        project_dir : str | Path
            Path to the project directory.

        Returns
        -------
        PRISMProject

        Raises
        ------
        FileNotFoundError
            If prism_project.json is missing.
        """
        proj = cls(project_dir)
        project_file = proj.root / PRISM_PROJECT_FILE
        if not project_file.exists():
            raise FileNotFoundError(
                f"Not a PRISM project: {proj.root}  "
                f"(missing {PRISM_PROJECT_FILE})"
            )

        data = json.loads(project_file.read_text())
        proj.metadata = data.get("metadata", proj.metadata)
        proj.spec = data.get("spec")
        proj.parameters = data.get("parameters", {})
        proj.notes = data.get("notes", [])
        proj.snapshots = [
            DesignSnapshot(**s) for s in data.get("snapshots", [])
        ]
        return proj

    # ── State management ──────────────────────────────────────────────

    def set_spec(self, spec) -> None:
        """Set the cage specification from a CageSpec or dict."""
        if hasattr(spec, "model_dump"):
            self.spec = spec.model_dump()
        elif isinstance(spec, dict):
            self.spec = spec
        else:
            raise TypeError(f"Expected CageSpec or dict, got {type(spec)}")

    def set_parameter(self, key: str, value: Any) -> None:
        """Store an arbitrary design parameter."""
        self.parameters[key] = value

    def add_note(self, note: str) -> None:
        """Add a timestamped note to the project."""
        self.notes.append(f"[{datetime.now().isoformat()}] {note}")

    def snapshot(
        self,
        description: str,
        metrics: Optional[dict] = None,
        files: Optional[list[str]] = None,
    ) -> DesignSnapshot:
        """Take a design snapshot.

        Parameters
        ----------
        description : str
            What this snapshot represents.
        metrics : dict, optional
            Quality metrics to store.
        files : list[str], optional
            Relative file paths associated with this snapshot.

        Returns
        -------
        DesignSnapshot
        """
        snap = DesignSnapshot(
            timestamp=datetime.now().isoformat(),
            description=description,
            spec=self.spec or {},
            metrics=metrics or {},
            files=files or [],
        )
        self.snapshots.append(snap)
        self.save()
        return snap

    # ── Persistence ───────────────────────────────────────────────────

    def save(self) -> Path:
        """Save project state to disk.

        Returns
        -------
        Path
            Path to the project JSON file.
        """
        self._ensure_dirs()
        data = {
            "metadata": self.metadata,
            "spec": self.spec,
            "parameters": self.parameters,
            "notes": self.notes,
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "description": s.description,
                    "spec": s.spec,
                    "metrics": s.metrics,
                    "files": s.files,
                }
                for s in self.snapshots
            ],
        }

        project_file = self.root / PRISM_PROJECT_FILE
        project_file.write_text(json.dumps(data, indent=2, default=str) + "\n")
        return project_file

    def export_archive(self, output_path: Optional[str | Path] = None) -> Path:
        """Export the project as a .tar.gz archive.

        Parameters
        ----------
        output_path : str | Path, optional
            Destination path. Default: project_name.tar.gz alongside project dir.

        Returns
        -------
        Path
        """
        if output_path is None:
            output_path = self.root.parent / f"{self.root.name}.tar.gz"
        output_path = Path(output_path)

        # Use shutil to create archive
        base_name = str(output_path).replace(".tar.gz", "")
        shutil.make_archive(base_name, "gztar", self.root.parent, self.root.name)
        return output_path

    # ── File helpers ──────────────────────────────────────────────────

    def store_structure(self, src_path: str | Path, name: Optional[str] = None) -> Path:
        """Copy a structure file into the project's structures/ directory.

        Parameters
        ----------
        src_path : str | Path
            Source PDB/mmCIF file.
        name : str, optional
            Destination filename. Default: original filename.

        Returns
        -------
        Path
            Path to the copied file.
        """
        src = Path(src_path)
        dst = self.structures_dir / (name or src.name)
        shutil.copy2(src, dst)
        return dst

    def store_design(self, src_path: str | Path, name: Optional[str] = None) -> Path:
        """Copy a design output file into designs/."""
        src = Path(src_path)
        dst = self.designs_dir / (name or src.name)
        shutil.copy2(src, dst)
        return dst

    def save_metrics(self, metrics: dict, filename: str = "metrics.json") -> Path:
        """Save quality metrics to the analysis/ directory."""
        dst = self.analysis_dir / filename
        dst.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
        return dst

    def list_structures(self) -> list[Path]:
        """List structure files in the project."""
        return sorted(self.structures_dir.glob("*")) if self.structures_dir.exists() else []

    def list_designs(self) -> list[Path]:
        """List design output files."""
        return sorted(self.designs_dir.glob("*")) if self.designs_dir.exists() else []

    # ── Display ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable project summary."""
        lines = [
            f"PRISM Project: {self.metadata.get('name', '?')}",
            f"  Location:   {self.root}",
            f"  Created:    {self.metadata.get('created', '?')}",
            f"  Version:    {self.metadata.get('version', '?')}",
            f"  Spec set:   {'Yes' if self.spec else 'No'}",
            f"  Snapshots:  {len(self.snapshots)}",
            f"  Structures: {len(self.list_structures())}",
            f"  Designs:    {len(self.list_designs())}",
            f"  Notes:      {len(self.notes)}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PRISMProject({self.root})"
