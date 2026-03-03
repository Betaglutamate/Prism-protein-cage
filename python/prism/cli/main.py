"""
PRISM CLI — command-line interface for protein cage design.

Usage::

    prism init my_project --symmetry T --diameter 10
    prism design subunit --project my_project
    prism sequence --project my_project
    prism interface --project my_project
    prism analyze --project my_project
    prism viz --project my_project --output cage.html
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="prism",
    help="PRISM — Programmable protein cage design for nanocrystal manufacturing",
    add_completion=False,
)
console = Console()


# ── init ──────────────────────────────────────────────────────────────

@app.command()
def init(
    project_dir: str = typer.Argument(..., help="Project directory path"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Project name"),
    symmetry: str = typer.Option("T", "--symmetry", "-s", help="Symmetry group: T, O, or I"),
    diameter: float = typer.Option(10.0, "--diameter", "-d", help="Target cavity diameter (nm)"),
    phase: str = typer.Option("Fe3O4_magnetite", "--phase", "-p", help="Target crystal phase"),
    subunit_length: str = typer.Option("80-120", "--subunit-length", help="Subunit length range"),
):
    """Initialise a new PRISM design project."""
    from prism.core.cage import CageSpec, CavitySpec, InterfaceSpec
    from prism.io.project import PRISMProject

    proj = PRISMProject.create(project_dir, name=name or Path(project_dir).stem)

    # Parse subunit length range
    parts = subunit_length.split("-")
    length_range = (int(parts[0]), int(parts[1]))

    spec = CageSpec(
        symmetry_group=symmetry,
        cavity=CavitySpec(
            target_diameter_nm=diameter,
            crystal_phase=phase,
        ),
        subunit_length_range=length_range,
    )
    proj.set_spec(spec)
    proj.add_note(f"Project initialised with {symmetry} symmetry, {diameter} nm cavity, {phase} phase")
    proj.save()

    console.print(Panel(
        f"[bold green]Project created:[/bold green] {proj.root}\n"
        f"Symmetry: {symmetry}  |  Diameter: {diameter} nm  |  Phase: {phase}",
        title="PRISM",
    ))


# ── design ────────────────────────────────────────────────────────────

@app.command()
def design(
    project: str = typer.Argument(..., help="Project directory"),
    rfdiffusion_dir: str = typer.Option(..., "--rfdiffusion-dir", help="Path to RFdiffusion install"),
    num_designs: int = typer.Option(10, "--num-designs", "-n", help="Number of backbones to sample"),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use GPU for inference"),
):
    """Design cage subunit backbones using RFdiffusion."""
    from prism.design.rfdiffusion import RFdiffusionRunner, RFdiffusionConfig
    from prism.io.project import PRISMProject

    proj = PRISMProject.load(project)
    if proj.spec is None:
        console.print("[red]Error:[/red] No cage spec set. Run `prism init` first.")
        raise typer.Exit(1)

    config = RFdiffusionConfig(
        rfdiffusion_dir=rfdiffusion_dir,
        output_dir=str(proj.designs_dir / "rfdiffusion"),
    )
    runner = RFdiffusionRunner(config)

    if not runner.check_installation():
        console.print("[red]Error:[/red] RFdiffusion not found. Check --rfdiffusion-dir.")
        raise typer.Exit(1)

    console.print(f"[bold]Designing {num_designs} cage subunits...[/bold]")

    sym = proj.spec.get("symmetry_group", "T")
    subunit_range = proj.spec.get("subunit_length_range", [80, 120])

    result = runner.design_cage_subunit(
        symmetry=sym,
        subunit_length=subunit_range,
        num_designs=num_designs,
    )

    for pdb_path in result.pdb_paths:
        proj.store_design(pdb_path)

    proj.add_note(f"Generated {len(result.pdb_paths)} backbone designs")
    proj.save()

    console.print(f"[green]✓[/green] {len(result.pdb_paths)} designs saved to {proj.designs_dir}")


# ── sequence ──────────────────────────────────────────────────────────

@app.command()
def sequence(
    project: str = typer.Argument(..., help="Project directory"),
    mpnn_dir: str = typer.Option(..., "--mpnn-dir", help="Path to ProteinMPNN install"),
    input_pdb: Optional[str] = typer.Option(None, "--input", "-i", help="Input PDB (default: latest design)"),
    num_seqs: int = typer.Option(8, "--num-seqs", "-n", help="Sequences per backbone"),
    temperature: float = typer.Option(0.1, "--temperature", "-t", help="Sampling temperature"),
):
    """Design sequences for a backbone using ProteinMPNN."""
    from prism.design.sequence import ProteinMPNNRunner, ProteinMPNNConfig
    from prism.io.project import PRISMProject

    proj = PRISMProject.load(project)

    if input_pdb is None:
        designs = proj.list_designs()
        pdbs = [p for p in designs if p.suffix == ".pdb"]
        if not pdbs:
            console.print("[red]No PDB designs found. Run `prism design` first.[/red]")
            raise typer.Exit(1)
        input_pdb = str(pdbs[-1])

    config = ProteinMPNNConfig(mpnn_dir=mpnn_dir)
    runner = ProteinMPNNRunner(config)

    console.print(f"[bold]Designing {num_seqs} sequences for {Path(input_pdb).name}...[/bold]")

    result = runner.design_sequences(
        pdb_path=input_pdb,
        num_sequences=num_seqs,
        temperature=temperature,
    )

    console.print(f"[green]✓[/green] {len(result.sequences)} sequences designed")
    for i, (seq, score) in enumerate(zip(result.sequences[:5], result.scores[:5])):
        console.print(f"  Seq {i + 1}: score={score:.3f}  {seq[:40]}...")


# ── interface ─────────────────────────────────────────────────────────

@app.command()
def interface(
    project: str = typer.Argument(..., help="Project directory"),
    bindcraft_dir: str = typer.Option(..., "--bindcraft-dir", help="Path to BindCraft install"),
    input_pdb: Optional[str] = typer.Option(None, "--input", "-i", help="Input cage PDB"),
    lattice: str = typer.Option("SC", "--lattice", "-l", help="Lattice type: SC, BCC, FCC, hex"),
):
    """Design docking interfaces for lattice assembly using BindCraft."""
    from prism.design.bindcraft import BindCraftRunner, BindCraftConfig
    from prism.io.project import PRISMProject

    proj = PRISMProject.load(project)

    if input_pdb is None:
        structures = proj.list_structures()
        pdbs = [p for p in structures if p.suffix == ".pdb"]
        if not pdbs:
            console.print("[red]No structures found. Design and expand a cage first.[/red]")
            raise typer.Exit(1)
        input_pdb = str(pdbs[-1])

    config = BindCraftConfig(bindcraft_dir=bindcraft_dir)
    runner = BindCraftRunner(config)

    console.print(f"[bold]Designing docking interfaces for {lattice} lattice...[/bold]")

    result = runner.design_docking_interface(
        cage_pdb=input_pdb,
        target_chain="A",
        hotspot_residues=[],  # Will be determined by exterior patch analysis
    )

    proj.add_note(f"Interface design for {lattice} lattice: {len(result.pdb_paths)} results")
    proj.save()

    console.print(f"[green]✓[/green] {len(result.pdb_paths)} interface designs generated")


# ── analyze ───────────────────────────────────────────────────────────

@app.command()
def analyze(
    project: str = typer.Argument(..., help="Project directory"),
    input_pdb: Optional[str] = typer.Option(None, "--input", "-i", help="Input assembled cage PDB"),
    target_diameter: Optional[float] = typer.Option(None, "--target-diameter", help="Target diameter (nm)"),
):
    """Analyse a cage design: cavity, clashes, quality metrics."""
    from prism.analysis.cavity_analysis import analyse_cavity
    from prism.analysis.clash_check import check_self_clashes
    from prism.core.structure import ProteinStructure
    from prism.io.project import PRISMProject

    proj = PRISMProject.load(project)

    if input_pdb is None:
        structures = proj.list_structures()
        pdbs = [p for p in structures if p.suffix == ".pdb"]
        if not pdbs:
            console.print("[red]No structures found.[/red]")
            raise typer.Exit(1)
        input_pdb = str(pdbs[-1])

    console.print(f"[bold]Analysing {Path(input_pdb).name}...[/bold]")

    struct = ProteinStructure.load(input_pdb)
    coords = struct.get_all_coords()
    ca = struct.get_ca_coords()
    cb = struct.get_cb_coords()

    # Target diameter from spec if not provided
    if target_diameter is None and proj.spec:
        cavity_spec = proj.spec.get("cavity", {})
        target_diameter = cavity_spec.get("target_diameter_nm")

    # Cavity analysis
    cavity = analyse_cavity(
        coords,
        ca_coords=ca,
        cb_coords=cb,
        target_diameter_nm=target_diameter,
    )
    console.print(Panel(cavity.summary(), title="Cavity Analysis"))

    # Clash analysis
    chain_ids = struct.get_per_atom_chain_ids()
    clashes = check_self_clashes(coords, chain_ids)
    console.print(Panel(clashes.summary(), title="Clash Analysis"))

    # Save metrics
    metrics = {
        "cavity_volume_nm3": cavity.volume_nm3,
        "effective_diameter_nm": cavity.effective_diameter_nm,
        "void_fraction": cavity.void_fraction,
        "n_clashes": clashes.n_clashes,
        "worst_overlap": clashes.worst_overlap,
    }
    proj.save_metrics(metrics)
    proj.add_note(f"Analysis complete: {cavity.effective_diameter_nm:.1f} nm cavity, {clashes.n_clashes} clashes")
    proj.save()

    console.print(f"[green]✓[/green] Results saved to {proj.analysis_dir}")


# ── viz ───────────────────────────────────────────────────────────────

@app.command()
def viz(
    input_pdb: str = typer.Argument(..., help="PDB file to visualise"),
    symmetry: Optional[str] = typer.Option(None, "--symmetry", "-s", help="Show symmetry axes"),
    style: str = typer.Option("cartoon", "--style", help="Rendering style"),
    width: int = typer.Option(800, "--width", "-w", help="Viewer width"),
    height: int = typer.Option(600, "--height", "-h", help="Viewer height"),
):
    """Launch an interactive 3D viewer for a cage structure."""
    from prism.viz.viewer import CageViewer

    v = CageViewer(width, height)
    v.show_full_cage(pdb_path=input_pdb, style=style)

    if symmetry:
        v.show_symmetry_axes(symmetry)

    console.print(f"[bold]Displaying {Path(input_pdb).name}[/bold]")
    v.show()


# ── info ──────────────────────────────────────────────────────────────

@app.command()
def info(
    project: str = typer.Argument(..., help="Project directory"),
):
    """Show project information and status."""
    from prism.io.project import PRISMProject

    proj = PRISMProject.load(project)

    table = Table(title=f"PRISM Project: {proj.metadata.get('name', '?')}")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Location", str(proj.root))
    table.add_row("Created", proj.metadata.get("created", "?"))
    table.add_row("Symmetry", proj.spec.get("symmetry_group", "?") if proj.spec else "Not set")

    if proj.spec and "cavity" in proj.spec:
        cav = proj.spec["cavity"]
        table.add_row("Diameter", f"{cav.get('target_diameter_nm', '?')} nm")
        table.add_row("Phase", cav.get("crystal_phase", "?"))

    table.add_row("Structures", str(len(proj.list_structures())))
    table.add_row("Designs", str(len(proj.list_designs())))
    table.add_row("Snapshots", str(len(proj.snapshots)))

    console.print(table)

    if proj.notes:
        console.print("\n[bold]Recent notes:[/bold]")
        for note in proj.notes[-5:]:
            console.print(f"  {note}")


# ── pipeline ──────────────────────────────────────────────────────────

@app.command()
def pipeline(
    project: str = typer.Argument(..., help="Project directory"),
    rfdiffusion_dir: str = typer.Option(..., "--rfdiffusion-dir", help="RFdiffusion path"),
    mpnn_dir: str = typer.Option(..., "--mpnn-dir", help="ProteinMPNN path"),
    num_designs: int = typer.Option(5, "--num-designs", help="Number of backbone designs"),
    num_seqs: int = typer.Option(4, "--num-seqs", help="Sequences per backbone"),
):
    """Run the full design pipeline: backbone → sequence → analysis."""
    console.print(Panel("[bold]PRISM Full Pipeline[/bold]", style="blue"))

    console.print("\n[bold cyan]Step 1/3:[/bold cyan] Backbone design")
    design(project=project, rfdiffusion_dir=rfdiffusion_dir, num_designs=num_designs)

    console.print("\n[bold cyan]Step 2/3:[/bold cyan] Sequence design")
    sequence(project=project, mpnn_dir=mpnn_dir, num_seqs=num_seqs)

    console.print("\n[bold cyan]Step 3/3:[/bold cyan] Analysis")
    analyze(project=project)

    console.print(Panel("[bold green]Pipeline complete![/bold green]", style="green"))


def main():
    """Entry point for the prism CLI."""
    app()


if __name__ == "__main__":
    main()
