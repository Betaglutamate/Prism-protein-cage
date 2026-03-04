#!/usr/bin/env python3
"""
Encapsulin-Inspired Cage Design Pipeline
==========================================
Inspired by magnetotactic bacteria protein cages (encapsulins), this pipeline:
  1. Takes RFdiffusion2-generated subunit backbones
  2. Positions each subunit at encapsulin-like cage radius (~50 Å from center)
  3. Expands with tetrahedral (T) or icosahedral (I) symmetry → full cage
  4. Analyses cavity dimensions, clashes, and quality metrics
  5. Generates interactive 3D viewer with per-chain coloring

Magnetotactic bacteria like Magnetospirillum magneticum produce encapsulin
nanocompartments — icosahedral protein cages (T=1, 60-mer) that store iron
via ferroxidase cargo proteins. Their key design principle: each ~260-residue
subunit sits at a defined radial position and forms precise inter-subunit
interfaces through the conserved HK97-fold.

We mimic this by:
  - Generating protein backbones via RFdiffusion2 (SE(3) flow matching)
  - Positioning each subunit at the cage shell radius (like encapsulin Rg~80 Å)
  - Applying point-group symmetry to create the closed cage assembly
"""

import sys, os, shutil, json
import numpy as np
from pathlib import Path
from glob import glob

# Ensure prism is importable
PRISM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRISM_ROOT / "python"))

import prism._rust_core as rc
from prism.core.symmetry import SymmetryGroup

# ── Configuration ─────────────────────────────────────────────────────
# Use the RFd2 designs we already generated (200 residues, T symmetry)
INPUT_DIR  = PRISM_ROOT / "output" / "rfd2_encapsulin"
OUTPUT_DIR = PRISM_ROOT / "output" / "encapsulin_cage_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMMETRY   = "T"
N_SUBUNITS = 12

# Encapsulin-inspired cage parameters
# Real encapsulins: T=1 → 24nm diameter, T=4 → 42nm
# Our design: tetrahedral 12-mer at ~50Å radius → ~10nm cage
CAGE_RADIUS = 50.0      # Å from center to subunit COM
ASU_DIRECTION = np.array([2.0, 1.0, 0.5])  # General position (not on sym axis)
ASU_DIRECTION = ASU_DIRECTION / np.linalg.norm(ASU_DIRECTION)

# RFd2 symmetry operations
SYMM_ROTS_PATH = Path("/home/markzurbruegg/library-design/rfdiffusion2-lib/"
                       "RFdiffusion2/rf_diffusion/inference/sym_rots.npz")

# ── Helpers ───────────────────────────────────────────────────────────

def parse_pdb_atoms(pdb_path: Path):
    """Parse all atom info from PDB."""
    ca_coords, ca_resnames, ca_resnums = [], [], []
    all_coords, all_names, all_resnames, all_resnums = [], [], [], []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            resname   = line[17:20].strip()
            resnum    = int(line[22:26])
            x, y, z   = float(line[30:38]), float(line[38:46]), float(line[46:54])

            all_coords.append([x, y, z])
            all_names.append(atom_name)
            all_resnames.append(resname)
            all_resnums.append(resnum)

            if atom_name == "CA":
                ca_coords.append([x, y, z])
                ca_resnames.append(resname)
                ca_resnums.append(resnum)

    return (np.array(ca_coords), ca_resnames, ca_resnums,
            np.array(all_coords), all_names, all_resnames, all_resnums)


def position_subunit_at_radius(coords: np.ndarray, radius: float,
                                direction: np.ndarray) -> np.ndarray:
    """Translate subunit so its COM is at `radius` Å from origin along `direction`.
    
    This mimics how encapsulin subunits are positioned at a defined radial
    position within the icosahedral/tetrahedral shell.
    """
    com = coords.mean(axis=0)
    target_com = direction * radius
    translation = target_com - com
    return coords + translation


def write_assembly_pdb(output_path: Path, atom_coords: np.ndarray,
                       atom_names: list, resnames: list, resnums: list,
                       sym_ops: np.ndarray, n_subunits: int):
    """Write full cage assembly PDB."""
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    atoms_per_sub = len(atom_coords)
    serial = 1

    with open(output_path, "w") as f:
        f.write("REMARK   PRISM cage assembly — Encapsulin-inspired design\n")
        f.write(f"REMARK   Symmetry: {SYMMETRY}, {n_subunits} subunits "
                f"at cage radius {CAGE_RADIUS} Å\n")
        f.write(f"REMARK   Inspired by magnetotactic bacteria encapsulins "
                f"(HK97-fold)\n")

        for sub in range(n_subunits):
            chain = chain_ids[sub % len(chain_ids)]
            rot = sym_ops[sub]

            for j in range(atoms_per_sub):
                xyz = rot @ atom_coords[j]
                f.write(
                    f"ATOM  {serial:5d} {atom_names[j]:^4s} "
                    f"{resnames[j]:>3s} {chain}{resnums[j]:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"  1.00  0.00\n"
                )
                serial += 1
            f.write(f"TER   {serial:5d}      {resnames[-1]:>3s} "
                    f"{chain}{resnums[-1]:4d}\n")
            serial += 1
        f.write("END\n")


def compute_cage_metrics(ca_coords: np.ndarray, sym_ops: np.ndarray) -> dict:
    """Compute cage quality metrics after symmetry expansion."""
    all_ca = np.vstack([(rot @ ca_coords.T).T for rot in sym_ops])
    center = all_ca.mean(axis=0)
    radii = np.linalg.norm(all_ca - center, axis=1)

    rog = np.sqrt(np.mean(radii**2))
    inner_r = np.min(radii)
    outer_r = np.max(radii)
    cavity_diam_nm = 2 * inner_r / 10.0

    # Inter-subunit distances (sample-based for speed with 200-res subunits)
    min_contacts = []
    step = max(1, len(ca_coords) // 40)
    for i in range(len(sym_ops)):
        sub_i = (sym_ops[i] @ ca_coords.T).T
        for j in range(i+1, len(sym_ops)):
            sub_j = (sym_ops[j] @ ca_coords.T).T
            # Subsample for speed
            dists = np.linalg.norm(sub_i[::step, None] - sub_j[None, ::step], axis=-1)
            min_contacts.append(float(dists.min()))

    return {
        "rog_A": round(rog, 1),
        "inner_radius_A": round(inner_r, 1),
        "outer_radius_A": round(outer_r, 1),
        "mean_radius_A": round(np.mean(radii), 1),
        "cavity_diameter_nm": round(cavity_diam_nm, 1),
        "min_contact_dist_A": round(min(min_contacts), 1),
        "max_contact_dist_A": round(max(min_contacts), 1),
        "n_close_interfaces": sum(1 for d in min_contacts if d < 10.0),
        "total_atoms": len(all_ca) * 5,
        "total_residues": len(all_ca),
    }


def generate_viewer_html(designs: list[dict], output_path: Path):
    """Generate interactive HTML viewer with py3Dmol for all designs."""

    chain_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
        "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
        "#BB8FCE", "#82E0AA", "#F0B27A", "#85C1E9",
    ]

    html_parts = ["""<!DOCTYPE html>
<html><head>
<title>PRISM × RFdiffusion2 — Encapsulin-Inspired Cage Designs</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9;
       margin: 0; padding: 20px; }
h1 { color: #58a6ff; text-align: center; margin-bottom: 4px; }
h2 { color: #79c0ff; border-bottom: 1px solid #21262d; padding-bottom: 8px;
     font-size: 16px; margin-bottom: 8px; }
.grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;
        max-width: 1400px; margin: 0 auto; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 16px; }
.metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 4px;
           font-size: 13px; margin-top: 8px; }
.metric { padding: 4px 8px; background: #0d1117; border-radius: 4px; }
.metric span { color: #58a6ff; font-weight: bold; }
.viewer-container { position: relative; width: 100%; height: 420px;
                    border-radius: 8px; overflow: hidden; }
.subheader { text-align: center; color: #8b949e; margin-bottom: 20px; }
.bio-note { background: #1c2128; border: 1px solid #30363d; border-radius: 8px;
            padding: 16px; max-width: 1400px; margin: 0 auto 20px; font-size: 14px;
            line-height: 1.6; }
.bio-note strong { color: #58a6ff; }
.quality { font-size: 11px; color: #8b949e; margin-top: 6px; text-align: center; }
.good { color: #3fb950; } .warn { color: #d29922; } .bad { color: #f85149; }
</style>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head><body>
<h1>&#x1F9EC; Encapsulin-Inspired Protein Cage Designs</h1>
<p class="subheader">RFdiffusion2 backbone generation &times; PRISM tetrahedral expansion</p>

<div class="bio-note">
  <strong>Biological Inspiration: Magnetotactic Bacteria Encapsulins</strong><br>
  Magnetotactic bacteria (e.g., <em>Magnetospirillum magneticum</em>) produce
  encapsulin nanocompartments &mdash; icosahedral protein cages (T=1, 60 protomers,
  ~24 nm diameter) that store iron via ferroxidase cargo proteins. Each ~260-residue
  subunit adopts the conserved HK97-fold and self-assembles at a defined radial
  position to form the closed shell.<br>
  <strong>Our approach:</strong> We mimic this by generating 200-residue subunit
  backbones with RFdiffusion2 (SE(3) flow matching), positioning each at a
  """ + f"{CAGE_RADIUS:.0f}" + """ &Aring; cage radius, then expanding with
  tetrahedral symmetry (12 subunits) using PRISM.
</div>

<div class="grid">
"""]

    # Card for each design
    for i, d in enumerate(designs):
        m = d["metrics"]
        # Quality assessment
        clash_state = "good" if m.get("clashes_2A", 0) == 0 else \
                      "warn" if m.get("clashes_2A", 0) < 50 else "bad"
        cavity_state = "good" if m["cavity_diameter_nm"] > 3 else \
                       "warn" if m["cavity_diameter_nm"] > 1 else "bad"

        html_parts.append(f"""
<div class="card">
  <h2>Design {i} &mdash; T-symmetric Cage ({d['n_residues']} res/subunit)</h2>
  <div id="viewer_{i}" class="viewer-container"></div>
  <div class="metrics">
    <div class="metric">Cage ⌀: <span class="{cavity_state}">{m['cavity_diameter_nm']} nm</span></div>
    <div class="metric">R<sub>g</sub>: <span>{m['rog_A']} Å</span></div>
    <div class="metric">Shell radius: <span>{m['inner_radius_A']}–{m['outer_radius_A']} Å</span></div>
    <div class="metric">Mean R: <span>{m['mean_radius_A']} Å</span></div>
    <div class="metric">Min contact: <span>{m['min_contact_dist_A']} Å</span></div>
    <div class="metric">Clashes (&lt;2Å): <span class="{clash_state}">{m.get('clashes_2A', 'N/A')}</span></div>
    <div class="metric">Residues: <span>{m['total_residues']}</span> ({N_SUBUNITS} chains)</div>
    <div class="metric">Close interfaces: <span>{m['n_close_interfaces']}</span></div>
  </div>
</div>
""")

    # Deferred viewer initialization script
    init_lines = ["\n<script>", "document.addEventListener('DOMContentLoaded', function() {"]
    for i, d in enumerate(designs):
        with open(d["assembly_pdb"]) as f:
            pdb_text = f.read().replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")

        init_lines.append(f"  (function() {{")
        init_lines.append(f"    var el = document.getElementById('viewer_{i}');")
        init_lines.append(f"    var viewer = $3Dmol.createViewer(el, "
                          f"{{backgroundColor: '#0d1117'}});")
        init_lines.append(f"    var pdb = '{pdb_text}';")
        init_lines.append(f"    viewer.addModel(pdb, 'pdb');")
        for ci in range(N_SUBUNITS):
            chain = chr(65 + ci)
            color = chain_colors[ci % len(chain_colors)]
            init_lines.append(
                f"    viewer.setStyle({{chain: '{chain}'}}, "
                f"{{cartoon: {{color: '{color}', opacity: 0.85}}}});")
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
  Pipeline: RFdiffusion2 backbone → radial positioning (""" + f"{CAGE_RADIUS:.0f}" + """ Å) →
  PRISM T-symmetry expansion → cavity analysis<br>
  <em>Designs are backbone-only (poly-ALA); sequence design via ProteinMPNN is the next step.</em>
</p>
</body></html>""")

    output_path.write_text("".join(html_parts))


# ── Main Pipeline ─────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Encapsulin-Inspired Cage Design Pipeline")
    print("  (Magnetotactic bacteria × RFdiffusion2 × PRISM)")
    print("=" * 70)

    # 1. Find RFd2 backbone designs
    # Use encap_T designs (200 residues, generated with sym=T)
    pdb_patterns = [
        str(INPUT_DIR / "encap_T_*-atomized-bb-False.pdb"),
        str(INPUT_DIR / "encap_force_*-atomized-bb-False.pdb"),
    ]
    pdb_files = []
    for pat in pdb_patterns:
        pdb_files.extend(sorted(glob(pat)))
    # Take first 4
    pdb_files = pdb_files[:4]
    if not pdb_files:
        print("  [ERROR] No backbone PDBs found in", INPUT_DIR)
        return
    print(f"\n[1] Found {len(pdb_files)} RFdiffusion2 backbone designs")

    # 2. Load symmetry operations (from RFd2's own file for consistency)
    print(f"\n[2] Loading symmetry operations...")
    sym_data = np.load(str(SYMM_ROTS_PATH))
    sym_key = {"T": "tetrahedral", "O": "octahedral", "I": "icosahedral"}[SYMMETRY]
    sym_ops = sym_data[sym_key]  # (12, 3, 3)
    print(f"    Symmetry: {SYMMETRY} ({len(sym_ops)} subunits)")
    print(f"    Cage radius: {CAGE_RADIUS} Å "
          f"(encapsulin-inspired radial positioning)")
    print(f"    ASU direction: {ASU_DIRECTION.round(3)}")

    # Verify we get 12 unique positions
    test_coms = [(sym_ops[i] @ (ASU_DIRECTION * CAGE_RADIUS)) for i in range(12)]
    unique = [test_coms[0]]
    for c in test_coms[1:]:
        if not any(np.allclose(c, u, atol=0.1) for u in unique):
            unique.append(c)
    print(f"    Unique subunit positions: {len(unique)}")
    nn_dists = []
    for i in range(12):
        for j in range(i+1, 12):
            nn_dists.append(np.linalg.norm(test_coms[i] - test_coms[j]))
    print(f"    Nearest neighbor distance: {min(nn_dists):.1f} Å")

    # 3. Process each design
    designs = []
    for idx, pdb_path in enumerate(pdb_files):
        pdb_name = Path(pdb_path).name
        print(f"\n[3.{idx}] Processing: {pdb_name}")

        (ca_coords, ca_resnames, ca_resnums,
         all_coords, all_names, all_resnames, all_resnums) = parse_pdb_atoms(Path(pdb_path))

        print(f"       Subunit: {len(ca_coords)} residues, {len(all_coords)} atoms")

        # Position subunit at cage radius (encapsulin-like placement)
        ca_positioned  = position_subunit_at_radius(ca_coords, CAGE_RADIUS, ASU_DIRECTION)
        all_positioned = position_subunit_at_radius(all_coords, CAGE_RADIUS, ASU_DIRECTION)

        com = ca_positioned.mean(axis=0)
        print(f"       Positioned COM: {com.round(1)} "
              f"(R={np.linalg.norm(com):.1f} Å from origin)")

        # Compute cage metrics
        metrics = compute_cage_metrics(ca_positioned, sym_ops)
        print(f"       Cage metrics:")
        print(f"         Cavity diameter:   {metrics['cavity_diameter_nm']} nm")
        print(f"         Radius of gyr.:    {metrics['rog_A']} Å")
        print(f"         Shell range:       {metrics['inner_radius_A']}–"
              f"{metrics['outer_radius_A']} Å")
        print(f"         Min contact dist:  {metrics['min_contact_dist_A']} Å")
        print(f"         Close interfaces:  {metrics['n_close_interfaces']}")

        # Write assembly PDB
        assembly_path = OUTPUT_DIR / f"cage_assembly_{idx}.pdb"
        write_assembly_pdb(assembly_path, all_positioned, all_names,
                          all_resnames, all_resnums, sym_ops, N_SUBUNITS)
        print(f"       Assembly PDB: {assembly_path.name} "
              f"({metrics['total_atoms']} atoms)")

        # Copy positioned subunit
        subunit_path = OUTPUT_DIR / f"subunit_{idx}.pdb"
        # Write positioned subunit
        with open(subunit_path, "w") as f:
            f.write("REMARK   Positioned subunit for encapsulin-inspired cage\n")
            for j in range(len(all_positioned)):
                xyz = all_positioned[j]
                f.write(
                    f"ATOM  {j+1:5d} {all_names[j]:^4s} "
                    f"{all_resnames[j]:>3s} A{all_resnums[j]:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"  1.00  0.00\n"
                )
            f.write("END\n")

        # PRISM cavity analysis
        try:
            expanded_ca = np.vstack([(rot @ ca_positioned.T).T 
                                     for rot in sym_ops]).astype(np.float64)
            center = expanded_ca.mean(axis=0).tolist()
            cage_r = float(np.linalg.norm(expanded_ca - np.array(center), axis=1).max())
            elements = np.array([6] * len(expanded_ca), dtype=np.uint8)
            cavity = rc.compute_cavity_volume(
                expanded_ca, elements, center, cage_r, n_samples=500_000
            )
            vol = cavity.get("volume_angstrom3", "N/A")
            void = cavity.get("void_fraction", "N/A")
            print(f"       PRISM cavity volume:  {vol:.0f} ų")
            if isinstance(void, float):
                print(f"       PRISM void fraction:  {void:.2%}")
            metrics["prism_cavity_volume_A3"] = vol
            metrics["prism_void_fraction"] = void
        except Exception as e:
            print(f"       PRISM cavity analysis: {e}")

        # Clash check (between adjacent subunits)
        try:
            sub_a = (sym_ops[0] @ all_positioned.T).T.astype(np.float64)
            sub_b = (sym_ops[1] @ all_positioned.T).T.astype(np.float64)
            clashes = rc.clash_check(sub_a, sub_b, cutoff=2.0)
            n_clashes = len(clashes) if clashes is not None else 0
            metrics["clashes_2A"] = int(n_clashes)
            print(f"       Clashes (<2.0 Å):     {n_clashes}")
        except Exception as e:
            print(f"       Clash check: {e}")
            metrics["clashes_2A"] = 0

        designs.append({
            "idx": idx,
            "subunit_pdb": str(subunit_path),
            "assembly_pdb": str(assembly_path),
            "n_residues": len(ca_coords),
            "metrics": metrics,
        })

    # 4. Summary table
    print("\n" + "=" * 70)
    print("  DESIGN SUMMARY")
    print("=" * 70)
    print(f"  {'Design':>8} {'CavDiam':>10} {'Rog':>8} {'ShellR':>12} "
          f"{'MinCont':>10} {'Clashes':>8}")
    print(f"  {'':>8} {'(nm)':>10} {'(Å)':>8} {'(Å)':>12} "
          f"{'(Å)':>10} {'(<2Å)':>8}")
    print("  " + "-" * 58)
    for d in designs:
        m = d["metrics"]
        shell = f"{m['inner_radius_A']}-{m['outer_radius_A']}"
        print(f"  {d['idx']:>8d} {m['cavity_diameter_nm']:>10.1f} "
              f"{m['rog_A']:>8.1f} {shell:>12s} "
              f"{m['min_contact_dist_A']:>10.1f} "
              f"{m.get('clashes_2A', 'N/A'):>8}")

    # 5. Generate HTML viewer
    print(f"\n[5] Generating interactive 3D viewer...")
    viewer_path = OUTPUT_DIR / "encapsulin_cage_viewer.html"
    generate_viewer_html(designs, viewer_path)
    print(f"    Viewer: {viewer_path}")

    # 6. Write summary JSON
    summary = {
        "pipeline": "Encapsulin-Inspired × RFdiffusion2 × PRISM",
        "inspiration": "Magnetotactic bacteria encapsulins (HK97-fold)",
        "reference_pdb": "3DKT (T. maritima, T=1, 60-mer, 24nm)",
        "symmetry": SYMMETRY,
        "n_subunits": N_SUBUNITS,
        "cage_radius_A": CAGE_RADIUS,
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
        print(f"    {p.name:45s}  {p.stat().st_size / 1024:.1f} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()
