#!/usr/bin/env python3
"""
RFdiffusion2 Cage Design Pipeline
===================================
Takes RFdiffusion2-generated subunit backbones (tetrahedral symmetry mode)
and processes them through the PRISM pipeline:
  1. Load RFd2 backbone PDBs
  2. Expand each with tetrahedral symmetry → full 12-subunit cage
  3. Analyse cavity dimensions, clashes, and metrics
  4. Generate interactive 3D viewer (HTML)
  5. Write assembly PDBs and FASTA sequences
"""

import sys, os
import numpy as np
from pathlib import Path
from glob import glob

# Ensure prism is importable
PRISM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRISM_ROOT / "python"))

import prism._rust_core as rc
from prism.core.symmetry import SymmetryGroup

# ── Configuration ─────────────────────────────────────────────────────
INPUT_DIR  = PRISM_ROOT / "output" / "rfd2_cage_designs"
OUTPUT_DIR = PRISM_ROOT / "output" / "rfd2_cage_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMMETRY   = "T"   # tetrahedral (12 subunits)
N_SUBUNITS = 12

# ── Helpers ───────────────────────────────────────────────────────────

def parse_pdb_atoms(pdb_path: Path) -> tuple[np.ndarray, list[str], list[int]]:
    """Parse CA coordinates, residue names, residue numbers from PDB."""
    coords = []
    resnames = []
    resnums = []
    all_coords = []    # all-atom
    all_names = []
    all_resnames = []
    all_resnums = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            resname   = line[17:20].strip()
            resnum    = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            all_coords.append([x, y, z])
            all_names.append(atom_name)
            all_resnames.append(resname)
            all_resnums.append(resnum)

            if atom_name == "CA":
                coords.append([x, y, z])
                resnames.append(resname)
                resnums.append(resnum)

    return (np.array(coords), resnames, resnums,
            np.array(all_coords), all_names, all_resnames, all_resnums)


def write_assembly_pdb(output_path: Path, all_atom_coords: np.ndarray,
                       all_atom_names: list, all_resnames: list,
                       all_resnums: list, sym_ops: np.ndarray,
                       n_subunits: int):
    """Write full cage assembly PDB by applying symmetry operations."""
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    atoms_per_subunit = len(all_atom_coords)
    atom_serial = 1

    with open(output_path, "w") as f:
        f.write(f"REMARK   PRISM cage assembly — RFdiffusion2 backbone\n")
        f.write(f"REMARK   Symmetry: {SYMMETRY}, subunits: {n_subunits}\n")
        f.write(f"REMARK   Residues/subunit: {max(all_resnums)}\n")

        for sub_i in range(n_subunits):
            chain = chain_ids[sub_i % len(chain_ids)]
            rot = sym_ops[sub_i]  # 3×3 rotation matrix

            for j in range(atoms_per_subunit):
                xyz = rot @ all_atom_coords[j]
                f.write(
                    f"ATOM  {atom_serial:5d} {all_atom_names[j]:^4s} "
                    f"{all_resnames[j]:>3s} {chain}{all_resnums[j]:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"  1.00  0.00\n"
                )
                atom_serial += 1

            f.write(f"TER   {atom_serial:5d}      {all_resnames[-1]:>3s} {chain}{all_resnums[-1]:4d}\n")
            atom_serial += 1

        f.write("END\n")


def compute_cage_metrics(ca_coords: np.ndarray, sym_ops: np.ndarray) -> dict:
    """Compute key cage metrics."""
    # Expand CA to all subunits
    all_ca = []
    for rot in sym_ops:
        all_ca.append((rot @ ca_coords.T).T)
    all_ca = np.vstack(all_ca)

    center = all_ca.mean(axis=0)

    # Radial distances from center
    radii = np.linalg.norm(all_ca - center, axis=1)

    # Radius of gyration
    rog = np.sqrt(np.mean(radii**2))

    # Inner/outer cavity radius estimates
    inner_radius = np.min(radii)
    outer_radius = np.max(radii)
    mean_radius = np.mean(radii)

    # Cavity diameter in nm
    cavity_diameter_nm = 2 * inner_radius / 10.0

    # Inter-subunit contacts: min distance between different subunits
    min_contacts = []
    for i in range(len(sym_ops)):
        for j in range(i+1, len(sym_ops)):
            sub_i = (sym_ops[i] @ ca_coords.T).T
            sub_j = (sym_ops[j] @ ca_coords.T).T
            dists = np.linalg.norm(sub_i[:, None] - sub_j[None, :], axis=-1)
            min_contacts.append(dists.min())

    return {
        "rog_A": round(float(rog), 1),
        "inner_radius_A": round(float(inner_radius), 1),
        "outer_radius_A": round(float(outer_radius), 1),
        "mean_radius_A": round(float(mean_radius), 1),
        "cavity_diameter_nm": round(float(cavity_diameter_nm), 1),
        "min_contact_dist_A": round(float(min(min_contacts)), 1),
        "max_contact_dist_A": round(float(max(min_contacts)), 1),
        "n_close_interfaces": sum(1 for d in min_contacts if d < 10.0),
        "total_atoms": len(all_ca) * 5,  # 5 atoms per residue
        "total_residues": len(all_ca),
    }


def generate_viewer_html(designs: list[dict], output_path: Path):
    """Generate interactive HTML viewer with all designs using py3Dmol."""
    try:
        import py3Dmol
    except ImportError:
        print("  [WARN] py3Dmol not installed, skipping HTML viewer")
        return

    html_parts = ["""<!DOCTYPE html>
<html><head>
<title>PRISM × RFdiffusion2 — Tetrahedral Cage Designs</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }
h1 { color: #58a6ff; text-align: center; }
h2 { color: #79c0ff; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
.grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; max-width: 1400px; margin: 0 auto; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 16px; }
.metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 13px; margin-top: 8px; }
.metric { padding: 4px 8px; background: #0d1117; border-radius: 4px; }
.metric span { color: #58a6ff; font-weight: bold; }
.viewer-container { position: relative; width: 100%; height: 400px; border-radius: 8px; overflow: hidden; }
.subheader { text-align: center; color: #8b949e; margin-bottom: 20px; }
</style>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head><body>
<h1>PRISM &times; RFdiffusion2 — Tetrahedral Cage Designs</h1>
<p class="subheader">Generated with RFdiffusion2 (sym.symid=T, 100 residues/subunit, 12 subunits)<br>
Model: RFD_140.pt | SE(3) flow matching | 50 denoising steps</p>
<div class="grid">
"""]

    chain_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
        "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
        "#BB8FCE", "#82E0AA", "#F0B27A", "#85C1E9",
    ]

    for i, d in enumerate(designs):
        metrics = d["metrics"]

        html_parts.append(f"""
<div class="card">
  <h2>Design {i} — Tetrahedral Cage ({d['n_residues']} res/subunit)</h2>
  <div id="viewer_{i}" class="viewer-container"></div>
  <div class="metrics">
    <div class="metric">Cavity ⌀: <span>{metrics['cavity_diameter_nm']} nm</span></div>
    <div class="metric">R<sub>g</sub>: <span>{metrics['rog_A']} Å</span></div>
    <div class="metric">Inner R: <span>{metrics['inner_radius_A']} Å</span></div>
    <div class="metric">Outer R: <span>{metrics['outer_radius_A']} Å</span></div>
    <div class="metric">Min contact: <span>{metrics['min_contact_dist_A']} Å</span></div>
    <div class="metric">Close interfaces: <span>{metrics['n_close_interfaces']}</span></div>
    <div class="metric">Total residues: <span>{metrics['total_residues']}</span></div>
    <div class="metric">Total atoms: <span>{metrics['total_atoms']}</span></div>
  </div>
</div>
""")

    # Build a single deferred initialization script for all viewers
    init_lines = ["<script>", "document.addEventListener('DOMContentLoaded', function() {"]
    for i, d in enumerate(designs):
        pdb_path = d["assembly_pdb"]
        with open(pdb_path) as f:
            pdb_text = f.read().replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")

        init_lines.append(f"  (function() {{")
        init_lines.append(f"    var el = document.getElementById('viewer_{i}');")
        init_lines.append(f"    var viewer = $3Dmol.createViewer(el, {{backgroundColor: '#0d1117'}});")
        init_lines.append(f"    var pdbData = '{pdb_text}';")
        init_lines.append(f"    viewer.addModel(pdbData, 'pdb');")
        for ci in range(N_SUBUNITS):
            chain = chr(65 + ci)
            color = chain_colors[ci % len(chain_colors)]
            init_lines.append(f"    viewer.setStyle({{chain: '{chain}'}}, {{cartoon: {{color: '{color}', opacity: 0.85}}}});")
        init_lines.append(f"    viewer.zoomTo();")
        init_lines.append(f"    viewer.render();")
        init_lines.append(f"    viewer.spin('y', 0.5);")
        init_lines.append(f"  }})();")

    init_lines.append("});")
    init_lines.append("</script>")
    html_parts.append("\n".join(init_lines))

    html_parts.append("""
</div>
<p class="subheader" style="margin-top: 30px;">
  Pipeline: RFdiffusion2 backbone → PRISM tetrahedral expansion → cavity analysis<br>
  <em>Designs are backbone-only (poly-ALA); sequence design via ProteinMPNN is the next step.</em>
</p>
</body></html>""")

    output_path.write_text("".join(html_parts))


# ── Main Pipeline ─────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  PRISM × RFdiffusion2 — Tetrahedral Cage Design Pipeline")
    print("=" * 70)

    # 1. Find RFd2-generated subunit PDBs
    pdb_files = sorted(glob(str(INPUT_DIR / "cage_T_*-atomized-bb-False.pdb")))
    if not pdb_files:
        print(f"[ERROR] No RFd2 backbone PDBs found in {INPUT_DIR}")
        sys.exit(1)
    print(f"\n[1] Found {len(pdb_files)} RFdiffusion2 backbone designs")

    # 2. Get tetrahedral symmetry operations
    print(f"\n[2] Loading tetrahedral symmetry operations...")
    sym = SymmetryGroup.from_name("T")
    sym_ops = sym.operations  # 12 × 3×3 rotation matrices

    print(f"    Symmetry: {SYMMETRY} ({len(sym_ops)} subunits)")

    # 3. Process each design
    designs = []
    for idx, pdb_path in enumerate(pdb_files):
        pdb_path = Path(pdb_path)
        print(f"\n[3.{idx}] Processing design {idx}: {pdb_path.name}")

        # Parse backbone
        ca_coords, resnames, resnums, all_coords, all_names, all_resnames, all_resnums = \
            parse_pdb_atoms(pdb_path)
        print(f"       Subunit: {len(ca_coords)} residues, {len(all_coords)} atoms")

        # Compute metrics
        metrics = compute_cage_metrics(ca_coords, sym_ops)
        print(f"       Cage metrics:")
        print(f"         Cavity diameter:   {metrics['cavity_diameter_nm']} nm")
        print(f"         Radius of gyr.:    {metrics['rog_A']} Å")
        print(f"         Inner / Outer R:   {metrics['inner_radius_A']} / {metrics['outer_radius_A']} Å")
        print(f"         Min contact dist:  {metrics['min_contact_dist_A']} Å")
        print(f"         Close interfaces:  {metrics['n_close_interfaces']}")

        # Write assembly PDB (12 chains)
        assembly_path = OUTPUT_DIR / f"cage_assembly_{idx}.pdb"
        write_assembly_pdb(assembly_path, all_coords, all_names,
                          all_resnames, all_resnums, sym_ops, N_SUBUNITS)
        print(f"       Assembly PDB: {assembly_path.name} "
              f"({metrics['total_atoms']} atoms, {N_SUBUNITS} chains)")

        # Copy subunit
        subunit_path = OUTPUT_DIR / f"subunit_{idx}.pdb"
        import shutil
        shutil.copy(pdb_path, subunit_path)

        # Expand all atoms to full cage
        expanded_all = []
        for rot in sym_ops:
            expanded_all.append((rot @ all_coords.T).T)
        expanded_all = np.vstack(expanded_all).astype(np.float64)

        expanded_ca = []
        for rot in sym_ops:
            expanded_ca.append((rot @ ca_coords.T).T)
        expanded_ca_arr = np.vstack(expanded_ca).astype(np.float64)

        # Run PRISM cavity analysis (Rust core)
        try:
            print(f"       Running PRISM cavity analysis...")
            center = expanded_ca_arr.mean(axis=0).tolist()
            cage_radius = float(np.linalg.norm(expanded_ca_arr - np.array(center), axis=1).max())
            # Build elements array (6=C for CA atoms); Rust expects u8 atomic numbers
            elements = np.array([6] * len(expanded_ca_arr), dtype=np.uint8)  # all C for CA-only

            cavity = rc.compute_cavity_volume(
                expanded_ca_arr, elements, center, cage_radius, n_samples=500_000
            )
            vol = cavity.get("volume_angstrom3", cavity.get("volume_A3", "N/A"))
            insc_r = cavity.get("inscribed_radius", "N/A")
            print(f"         PRISM cavity volume:  {vol} ų")
            print(f"         PRISM inscribed R:    {insc_r} Å")
            metrics["prism_cavity_volume_A3"] = vol
            metrics["prism_inscribed_radius_A"] = insc_r
        except Exception as e:
            print(f"         PRISM cavity analysis: {e}")

        # Run PRISM clash check (between different subunits)
        try:
            sub_a = (sym_ops[0] @ all_coords.T).T.astype(np.float64)
            sub_b = (sym_ops[1] @ all_coords.T).T.astype(np.float64)
            clashes = rc.clash_check(sub_a, sub_b, cutoff=2.0)
            n_clashes = len(clashes) if clashes is not None else 0
            metrics["clashes_2A"] = int(n_clashes)
            print(f"         Clashes (< 2.0 Å):    {n_clashes}")
        except Exception as e:
            print(f"         Clash check: {e}")

        designs.append({
            "idx": idx,
            "subunit_pdb": str(subunit_path),
            "assembly_pdb": str(assembly_path),
            "metrics": metrics,
            "n_residues": len(ca_coords),
        })

    # 4. Summary table
    print("\n" + "=" * 70)
    print("  DESIGN SUMMARY")
    print("=" * 70)
    print(f"  {'Design':>8} {'CavDiam':>10} {'Rog':>8} {'MinCont':>10} {'Ifaces':>8}")
    print(f"  {'':>8} {'(nm)':>10} {'(Å)':>8} {'(Å)':>10} {'':>8}")
    print("  " + "-" * 50)
    for d in designs:
        m = d["metrics"]
        print(f"  {d['idx']:>8d} {m['cavity_diameter_nm']:>10.1f} "
              f"{m['rog_A']:>8.1f} {m['min_contact_dist_A']:>10.1f} "
              f"{m['n_close_interfaces']:>8d}")

    # 5. Generate HTML viewer
    print(f"\n[5] Generating interactive 3D viewer...")
    viewer_path = OUTPUT_DIR / "rfd2_cage_viewer.html"
    generate_viewer_html(designs, viewer_path)
    print(f"    Viewer: {viewer_path}")

    # 6. Write summary JSON
    import json
    summary = {
        "pipeline": "PRISM × RFdiffusion2",
        "symmetry": SYMMETRY,
        "n_subunits": N_SUBUNITS,
        "model": "RFD_140.pt",
        "n_designs": len(designs),
        "designs": [
            {
                "idx": d["idx"],
                "subunit_pdb": d["subunit_pdb"],
                "assembly_pdb": d["assembly_pdb"],
                "n_residues": d["n_residues"],
                "metrics": d["metrics"],
            }
            for d in designs
        ],
    }
    summary_path = OUTPUT_DIR / "design_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    Summary: {summary_path}")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files:")
    for p in sorted(OUTPUT_DIR.glob("*")):
        print(f"    {p.name:40s}  {p.stat().st_size / 1024:.1f} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()
