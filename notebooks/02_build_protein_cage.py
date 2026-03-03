#!/usr/bin/env python3
"""
PRISM — Build a Full Protein Cage with 3D Visualization & Sequences
====================================================================

This script:
1. Generates a realistic α-helical bundle subunit (backbone: N, CA, C, O)
2. Defines a cage specification (tetrahedral, 10 nm, Fe₃O₄ magnetite)
3. Expands the subunit via symmetry → full 12-subunit cage assembly
4. Writes multi-chain PDB files (subunit + full cage)
5. Assigns interior surface chemistry and identifies interior residues
6. Generates designed sequences with residue-appropriate composition
7. Writes FASTA output
8. Produces an interactive 3D HTML viewer (py3Dmol)
9. Runs cavity analysis & clash detection on the full assembly
"""

import os
import numpy as np
from pathlib import Path

# ── Output directory ─────────────────────────────────────────────────
OUTPUT_DIR = Path("output/cage_design")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  PRISM — Protein Cage Builder")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
# 1. GENERATE A REALISTIC α-HELICAL BUNDLE SUBUNIT
# ═══════════════════════════════════════════════════════════════════════
print("\n[1/9] Generating α-helical bundle subunit backbone...")

# Standard backbone geometry parameters
BOND_N_CA  = 1.458   # N-Cα bond length (Å)
BOND_CA_C  = 1.523   # Cα-C bond length (Å)
BOND_C_N   = 1.329   # C-N (peptide bond) length (Å)
BOND_C_O   = 1.231   # C=O bond length (Å)

# α-helix parameters
HELIX_RISE = 1.5     # Rise per residue along helix axis (Å)
HELIX_RADIUS = 2.3   # Radius of Cα helix (Å)
HELIX_TWIST = np.radians(100)  # Twist per residue

N_RESIDUES = 100  # residues per subunit
N_HELICES = 4     # 4-helix bundle

def generate_alpha_helix(n_residues, center, axis_direction, start_phase=0.0):
    """Generate backbone atoms (N, CA, C, O) for an ideal α-helix."""
    axis = np.array(axis_direction, dtype=float)
    axis /= np.linalg.norm(axis)

    # Find two perpendicular directions
    if abs(axis[2]) < 0.9:
        perp1 = np.cross(axis, [0, 0, 1])
    else:
        perp1 = np.cross(axis, [1, 0, 0])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    coords = []
    residue_names = []
    atom_names = []
    elements = []

    for i in range(n_residues):
        phase = start_phase + i * HELIX_TWIST
        rise = i * HELIX_RISE

        # Cα position on helix
        ca = (center + rise * axis
              + HELIX_RADIUS * np.cos(phase) * perp1
              + HELIX_RADIUS * np.sin(phase) * perp2)

        # N position (slightly before CA along helix)
        n_phase = phase - 0.3
        n_rise = rise - 0.5
        n = (center + n_rise * axis
             + (HELIX_RADIUS - 0.3) * np.cos(n_phase) * perp1
             + (HELIX_RADIUS - 0.3) * np.sin(n_phase) * perp2)

        # C position (slightly after CA)
        c_phase = phase + 0.3
        c_rise = rise + 0.5
        c = (center + c_rise * axis
             + (HELIX_RADIUS + 0.2) * np.cos(c_phase) * perp1
             + (HELIX_RADIUS + 0.2) * np.sin(c_phase) * perp2)

        # O position (perpendicular to C, roughly pointing outward)
        o_dir = c - ca
        o_dir /= np.linalg.norm(o_dir)
        o_perp = np.cross(o_dir, axis)
        if np.linalg.norm(o_perp) > 0.01:
            o_perp /= np.linalg.norm(o_perp)
        o = c + BOND_C_O * o_perp

        for atom_coord, aname, elem in [(n, "N", "N"), (ca, "CA", "C"),
                                         (c, "C", "C"), (o, "O", "O")]:
            coords.append(atom_coord)
            atom_names.append(aname)
            elements.append(elem)
        residue_names.append("ALA")  # placeholder, will assign later

    return np.array(coords), residue_names, atom_names, elements


# Generate a 4-helix bundle positioned at cage radius (~50 Å from origin)
CAGE_RADIUS = 50.0  # Å, gives ~10 nm diameter cavity

bundle_coords = []
bundle_res_names = []
bundle_atom_names = []
bundle_elements = []

helix_length = N_RESIDUES // N_HELICES  # 25 residues per helix
bundle_radius = 5.0  # radius of the helix bundle cross-section

for h in range(N_HELICES):
    angle = h * 2 * np.pi / N_HELICES
    # Center each helix in the bundle, positioned at cage radius
    helix_center = np.array([
        CAGE_RADIUS + bundle_radius * np.cos(angle),
        bundle_radius * np.sin(angle),
        -helix_length * HELIX_RISE / 2  # center vertically
    ])
    # Helix axis along z
    direction = [0, 0, 1]
    phase = angle  # stagger helices

    coords, res_names, atom_names, elems = generate_alpha_helix(
        helix_length, helix_center, direction, start_phase=phase
    )
    bundle_coords.append(coords)
    bundle_res_names.extend(res_names)
    bundle_atom_names.extend(atom_names)
    bundle_elements.extend(elems)

subunit_coords = np.vstack(bundle_coords)
n_atoms_subunit = len(subunit_coords)
n_residues_total = N_RESIDUES

print(f"  Subunit: {N_HELICES}-helix bundle, {n_residues_total} residues, "
      f"{n_atoms_subunit} atoms")
print(f"  Positioned at radius {CAGE_RADIUS:.0f} Å from origin")

# ═══════════════════════════════════════════════════════════════════════
# 2. CAGE SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════
print("\n[2/9] Creating cage specification...")

from prism.core.cage import CageSpec, CavitySpec, CageDesign

spec = CageSpec(
    name="Fe3O4_T_cage_v1",
    symmetry_group="T",
    cavity=CavitySpec(
        target_diameter_nm=10.0,
        crystal_phase="Fe3O4_magnetite",
        shape="spherical",
    ),
    subunit_length_range=(80, 120),
    lattice_type="SC",
)

cage = CageDesign(spec=spec)
cage.set_subunit(subunit_coords)
cage.set_surface_chemistry("Fe3O4_magnetite")

print(f"  {cage.summary()}")

# ═══════════════════════════════════════════════════════════════════════
# 3. SYMMETRY EXPANSION → FULL CAGE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════
print("\n[3/9] Expanding subunit via tetrahedral symmetry...")

from prism.core.symmetry import SymmetryGroup

sg = SymmetryGroup.from_name("T")
assembly_coords = cage.expand()

# Get chain IDs for the expanded structure
_, chain_ids = sg.expand_with_chain_ids(subunit_coords)
unique_chains = sorted(set(chain_ids))

print(f"  Symmetry group: T ({sg.order} operations)")
print(f"  Assembly: {len(assembly_coords)} atoms across {len(unique_chains)} chains")
print(f"  Cage diameter: ~{2 * np.max(np.linalg.norm(assembly_coords, axis=1)):.0f} Å "
      f"({2 * np.max(np.linalg.norm(assembly_coords, axis=1)) / 10:.1f} nm)")

# ═══════════════════════════════════════════════════════════════════════
# 4. WRITE PDB FILES
# ═══════════════════════════════════════════════════════════════════════
print("\n[4/9] Writing PDB files...")

from prism.io.pdb import write_pdb

# -- Subunit PDB --
subunit_chain_ids = np.array(["A"] * n_atoms_subunit)
# Proper residue numbering: 4 atoms per residue
subunit_residue_names = []
for rn in bundle_res_names:
    subunit_residue_names.extend([rn] * 4)  # N, CA, C, O per residue

subunit_pdb = write_pdb(
    subunit_coords,
    OUTPUT_DIR / "subunit.pdb",
    chain_ids=subunit_chain_ids,
    residue_names=subunit_residue_names,
    atom_names=bundle_atom_names,
    elements=bundle_elements,
)
print(f"  Subunit PDB: {subunit_pdb}")

# -- Full cage PDB (multi-chain) --
assembly_res_names = []
assembly_atom_names = []
assembly_elements = []
assembly_chain_arr = []

for i, ch in enumerate(unique_chains):
    assembly_res_names.extend(subunit_residue_names)
    assembly_atom_names.extend(bundle_atom_names)
    assembly_elements.extend(bundle_elements)
    assembly_chain_arr.extend([ch] * n_atoms_subunit)

assembly_chain_arr = np.array(assembly_chain_arr)

cage_pdb = write_pdb(
    assembly_coords,
    OUTPUT_DIR / "cage_assembly.pdb",
    chain_ids=assembly_chain_arr,
    residue_names=assembly_res_names,
    atom_names=assembly_atom_names,
    elements=assembly_elements,
)
print(f"  Full cage PDB: {cage_pdb} ({len(assembly_coords)} atoms, {len(unique_chains)} chains)")

# ═══════════════════════════════════════════════════════════════════════
# 5. SURFACE CHEMISTRY — IDENTIFY INTERIOR RESIDUES
# ═══════════════════════════════════════════════════════════════════════
print("\n[5/9] Assigning interior surface chemistry...")

from prism.core.residue_surface import SurfaceChemSpec

surf_chem = SurfaceChemSpec.for_phase("Fe3O4_magnetite")
print(f"  Target phase: {surf_chem.target_phase}")
print(f"  Coordination geometry: {surf_chem.coordination_geometry}")
print(f"  Preferred interior residues: {surf_chem.preferred_residues}")

# Identify interior-facing Cα atoms (those pointing toward cage center)
ca_mask = np.array([n == "CA" for n in bundle_atom_names])
ca_coords = subunit_coords[ca_mask]
center = np.array([0.0, 0.0, 0.0])

# Interior = Cα atoms whose radial vector points inward
interior_mask = []
for i, ca in enumerate(ca_coords):
    # Is the Cα on the inner face of the bundle?
    dist_to_center = np.linalg.norm(ca - center)
    interior_mask.append(dist_to_center < CAGE_RADIUS)

interior_indices = np.where(interior_mask)[0]
n_interior = len(interior_indices)
print(f"  Interior-facing residues: {n_interior}/{len(ca_coords)}")

# ═══════════════════════════════════════════════════════════════════════
# 6. GENERATE DESIGNED SEQUENCES
# ═══════════════════════════════════════════════════════════════════════
print("\n[6/9] Designing sequences...")

# Amino acid frequencies for a stable helical protein
HELIX_AA_FREQ = {
    'A': 0.18, 'L': 0.15, 'E': 0.12, 'K': 0.10, 'R': 0.08,
    'Q': 0.07, 'I': 0.06, 'V': 0.05, 'M': 0.04, 'F': 0.03,
    'D': 0.03, 'N': 0.03, 'S': 0.02, 'T': 0.02, 'Y': 0.01,
    'W': 0.01,
}
# Interior surface residues (for Fe3O4 coordination)
INTERIOR_AA = {'H': 0.35, 'E': 0.25, 'D': 0.25, 'C': 0.15}

np.random.seed(42)
aa_pool = list(HELIX_AA_FREQ.keys())
aa_weights = np.array(list(HELIX_AA_FREQ.values()))
aa_weights /= aa_weights.sum()

interior_pool = list(INTERIOR_AA.keys())
interior_weights = np.array(list(INTERIOR_AA.values()))
interior_weights /= interior_weights.sum()

N_DESIGNS = 4
sequences = {}
design_scores = []

for design_id in range(N_DESIGNS):
    seq = []
    for res_idx in range(n_residues_total):
        if res_idx in interior_indices:
            # Use Fe3O4-coordinating residues for interior positions
            aa = np.random.choice(interior_pool, p=interior_weights)
        else:
            # Use helix-favorable residues for the rest
            aa = np.random.choice(aa_pool, p=aa_weights)
        seq.append(aa)

    sequence = "".join(seq)
    sequences[f"Fe3O4_T_cage_design_{design_id+1}"] = sequence

    # Compute a mock stability score (fraction helix-favorable + interior match)
    helix_frac = sum(1 for aa in seq if aa in 'ALER') / len(seq)
    interior_frac = sum(1 for i, aa in enumerate(seq)
                       if i in interior_indices and aa in 'HEDC') / max(n_interior, 1)
    score = -(helix_frac * 0.6 + interior_frac * 0.4) * 3.0  # lower = better
    design_scores.append(score)

print(f"  Generated {N_DESIGNS} sequence designs")
print(f"  Subunit length: {n_residues_total} residues")

# Display sequences
print("\n  ┌─ Designed Sequences ─────────────────────────────────────────┐")
for name, seq in sequences.items():
    score = design_scores[list(sequences.keys()).index(name)]
    print(f"  │ {name}")
    print(f"  │   Score: {score:.3f}")
    print(f"  │   Seq:   {seq[:60]}...")
    helix_content = sum(1 for aa in seq if aa in 'AELKR') / len(seq)
    print(f"  │   Helix content: {helix_content:.1%}")
    interior_res = [seq[i] for i in interior_indices if i < len(seq)]
    print(f"  │   Interior residues: {''.join(interior_res[:20])}...")
    print(f"  │")
print(f"  └─────────────────────────────────────────────────────────────┘")

best_idx = np.argmin(design_scores)
best_name = list(sequences.keys())[best_idx]
print(f"\n  Best design: {best_name} (score: {design_scores[best_idx]:.3f})")

# ═══════════════════════════════════════════════════════════════════════
# 7. WRITE FASTA OUTPUT
# ═══════════════════════════════════════════════════════════════════════
print("\n[7/9] Writing FASTA output...")

from prism.io.pdb import write_fasta

fasta_path = write_fasta(sequences, OUTPUT_DIR / "designed_sequences.fasta")
print(f"  FASTA file: {fasta_path}")

# Also write per-chain FASTA for the full cage assembly
cage_sequences = {}
best_seq = sequences[best_name]
for i, ch in enumerate(unique_chains):
    cage_sequences[f"chain_{ch}_subunit_{i+1}"] = best_seq

cage_fasta = write_fasta(cage_sequences, OUTPUT_DIR / "cage_assembly_sequences.fasta")
print(f"  Cage assembly FASTA: {cage_fasta} ({len(cage_sequences)} chains)")

# ═══════════════════════════════════════════════════════════════════════
# 8. 3D VISUALIZATION (HTML output)
# ═══════════════════════════════════════════════════════════════════════
print("\n[8/9] Building 3D visualization...")

import py3Dmol

# -- Subunit view --
subunit_pdb_data = Path(subunit_pdb).read_text()
cage_pdb_data = Path(cage_pdb).read_text()

CHAIN_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]

# Build an HTML file with two views
html_parts = ["""<!DOCTYPE html>
<html>
<head>
    <title>PRISM Cage Design — 3D Viewer</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 20px;
               background: #1a1a2e; color: #e0e0e0; }
        h1 { color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }
        h2 { color: #00d4ff; margin-top: 30px; }
        .viewer-container { display: flex; gap: 20px; flex-wrap: wrap;
                           justify-content: center; margin: 20px 0; }
        .viewer-box { background: #16213e; border-radius: 12px; padding: 15px;
                     box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1); }
        .viewer-label { text-align: center; font-weight: bold; margin-bottom: 10px;
                       color: #00d4ff; font-size: 14px; }
        .info-box { background: #16213e; border-radius: 8px; padding: 15px;
                   margin: 10px 0; border-left: 4px solid #00d4ff; }
        .seq-box { background: #0f3460; border-radius: 8px; padding: 15px;
                  margin: 10px 0; font-family: 'Courier New', monospace;
                  font-size: 12px; word-break: break-all; line-height: 1.6; }
        .seq-header { font-weight: bold; color: #00d4ff; margin-bottom: 5px;
                     font-family: Arial, sans-serif; }
        .interior { color: #ff6b6b; font-weight: bold; }
        .helix { color: #51cf66; }
        table { border-collapse: collapse; margin: 10px 0; }
        td, th { padding: 8px 12px; border: 1px solid #333; }
        th { background: #0f3460; color: #00d4ff; }
    </style>
</head>
<body>
    <h1>🧬 PRISM — Protein Cage Design Output</h1>

    <div class="info-box">
        <strong>Design:</strong> Fe₃O₄ Magnetite Templating Cage<br>
        <strong>Symmetry:</strong> Tetrahedral (T) — 12 subunits<br>
        <strong>Target cavity:</strong> 10.0 nm diameter<br>
        <strong>Crystal phase:</strong> Fe₃O₄ magnetite<br>
"""]

html_parts.append(f"""
        <strong>Subunit:</strong> {n_residues_total} residues, {n_atoms_subunit} atoms ({N_HELICES}-helix bundle)<br>
        <strong>Assembly:</strong> {len(assembly_coords)} atoms, {len(unique_chains)} chains<br>
        <strong>Interior residues:</strong> {n_interior} (HIS, GLU, ASP, CYS for Fe₃O₄ coordination)
    </div>

    <h2>3D Structure Views</h2>
    <div class="viewer-container">
        <div class="viewer-box">
            <div class="viewer-label">Single Subunit (Chain A)</div>
            <div id="subunit-viewer" style="width: 500px; height: 400px;"></div>
        </div>
        <div class="viewer-box">
            <div class="viewer-label">Full Cage Assembly (12 subunits)</div>
            <div id="cage-viewer" style="width: 500px; height: 400px;"></div>
        </div>
    </div>
    <div class="viewer-container">
        <div class="viewer-box">
            <div class="viewer-label">Cage with Symmetry Axes</div>
            <div id="axes-viewer" style="width: 500px; height: 400px;"></div>
        </div>
        <div class="viewer-box">
            <div class="viewer-label">Cage Surface (Cavity View)</div>
            <div id="surface-viewer" style="width: 500px; height: 400px;"></div>
        </div>
    </div>
""")

# Escape PDB data for JavaScript
import json
subunit_js = json.dumps(subunit_pdb_data)
cage_js = json.dumps(cage_pdb_data)

# Build symmetry axes data for JavaScript
axes_data = sg.get_symmetry_axes()
axes_js_lines = []
axis_colours = {"C2": "#ff4444", "C3": "#44ff44", "C5": "#4444ff"}

for order_key, axis_list in axes_data.items():
    col = axis_colours.get(order_key, "#888888")
    for ax in axis_list:
        ax = np.array(ax, dtype=float)
        ax = ax / np.linalg.norm(ax)
        start = -60 * ax
        end = 60 * ax
        axes_js_lines.append(
            f'v3.addCylinder({{start:{{x:{start[0]:.2f},y:{start[1]:.2f},z:{start[2]:.2f}}},'
            f'end:{{x:{end[0]:.2f},y:{end[1]:.2f},z:{end[2]:.2f}}},'
            f'radius:0.5,fromCap:true,toCap:true,color:"{col}"}});'
        )

html_parts.append(f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        // Subunit viewer
        var v1 = $3Dmol.createViewer('subunit-viewer', {{backgroundColor: '#1a1a2e'}});
        v1.addModel({subunit_js}, 'pdb');
        v1.setStyle({{}}, {{cartoon: {{color: '#1f77b4'}}, stick: {{radius: 0.15, color: '#1f77b4'}}}});
        v1.zoomTo();
        v1.render();

        // Full cage viewer
        var v2 = $3Dmol.createViewer('cage-viewer', {{backgroundColor: '#1a1a2e'}});
        v2.addModel({cage_js}, 'pdb');
""")

# Color each chain differently
for i, ch in enumerate(unique_chains):
    col = CHAIN_COLOURS[i % len(CHAIN_COLOURS)]
    html_parts.append(
        f'        v2.setStyle({{chain: "{ch}"}}, {{cartoon: {{color: "{col}"}}}});'
    )

html_parts.append("""
        v2.zoomTo();
        v2.render();

        // Axes viewer
        var v3 = $3Dmol.createViewer('axes-viewer', {backgroundColor: '#1a1a2e'});
        v3.addModel(""" + cage_js + """, 'pdb');
""")

for i, ch in enumerate(unique_chains):
    col = CHAIN_COLOURS[i % len(CHAIN_COLOURS)]
    html_parts.append(
        f'        v3.setStyle({{chain: "{ch}"}}, {{cartoon: {{color: "{col}", opacity: 0.6}}}});'
    )

for line in axes_js_lines:
    html_parts.append(f"        {line}")

html_parts.append("""
        v3.zoomTo();
        v3.render();

        // Surface viewer
        var v4 = $3Dmol.createViewer('surface-viewer', {backgroundColor: '#1a1a2e'});
        v4.addModel(""" + cage_js + """, 'pdb');
        v4.setStyle({}, {cartoon: {opacity: 0.4, color: '#888888'}});
        v4.addSurface('VDW', {opacity: 0.25, color: '#ff6666'}, {});
        v4.zoomTo();
        v4.render();
    });
    </script>
""")

# Add sequence section
html_parts.append("""
    <h2>Designed Sequences</h2>
    <table>
        <tr><th>Design</th><th>Score</th><th>Length</th><th>Helix %</th><th>Interior Match</th></tr>
""")

for i, (name, seq) in enumerate(sequences.items()):
    score = design_scores[i]
    helix_pct = sum(1 for aa in seq if aa in 'AELKR') / len(seq) * 100
    interior_res = [seq[j] for j in interior_indices if j < len(seq)]
    int_match = sum(1 for aa in interior_res if aa in 'HEDC') / max(len(interior_res), 1) * 100
    best_marker = " ⭐" if i == best_idx else ""
    html_parts.append(
        f'        <tr><td>{name}{best_marker}</td><td>{score:.3f}</td>'
        f'<td>{len(seq)}</td><td>{helix_pct:.1f}%</td><td>{int_match:.1f}%</td></tr>'
    )

html_parts.append("    </table>")

# Show best sequence with colored residues
best_seq = sequences[best_name]
html_parts.append(f"""
    <h2>Best Design: {best_name}</h2>
    <div class="seq-box">
        <div class="seq-header">Sequence ({len(best_seq)} residues):</div>
""")

# Color-code the sequence
colored_seq = []
for i, aa in enumerate(best_seq):
    if i in interior_indices:
        colored_seq.append(f'<span class="interior">{aa}</span>')
    elif aa in 'AELKR':
        colored_seq.append(f'<span class="helix">{aa}</span>')
    else:
        colored_seq.append(aa)

# Break into lines of 50
for start in range(0, len(colored_seq), 50):
    chunk = "".join(colored_seq[start:start+50])
    pos_label = f"{start+1:>4d}"
    html_parts.append(f"        {pos_label}  {chunk}<br>")

html_parts.append("""
        <br>
        <span class="interior">■</span> Interior (Fe₃O₄ coordinating: H, E, D, C) &nbsp;
        <span class="helix">■</span> Helix-favorable (A, E, L, K, R) &nbsp;
        ■ Other
    </div>
""")

html_parts.append("</body></html>")

html_content = "\n".join(html_parts)
html_path = OUTPUT_DIR / "cage_viewer.html"
html_path.write_text(html_content)
print(f"  3D viewer HTML: {html_path}")
print(f"  (Open in browser for interactive 3D views)")

# ═══════════════════════════════════════════════════════════════════════
# 9. ANALYSIS — CAVITY & CLASHES
# ═══════════════════════════════════════════════════════════════════════
print("\n[9/9] Running structural analysis...")

from prism.analysis.cavity_analysis import analyse_cavity
from prism.analysis.clash_check import check_clashes
from prism.analysis.metrics import symmetry_rmsd, packing_density

# Cavity analysis
cavity_report = analyse_cavity(
    assembly_coords,
    target_diameter_nm=10.0,
    n_samples=200_000,
)
print(f"\n  Cavity Analysis:")
print(f"    {cavity_report.summary()}")

# Clash check between adjacent subunits
chain_A_mask = assembly_chain_arr == unique_chains[0]
chain_B_mask = assembly_chain_arr == unique_chains[1]
coords_A = assembly_coords[chain_A_mask]
coords_B = assembly_coords[chain_B_mask]

clash_report = check_clashes(coords_A, coords_B, cutoff=2.0)
print(f"\n  Inter-subunit clash check (A vs B):")
print(f"    {clash_report.summary()}")

# Symmetry RMSD
subunits_for_rmsd = []
for ch in unique_chains[:4]:
    mask = assembly_chain_arr == ch
    subunits_for_rmsd.append(assembly_coords[mask])

rmsd = symmetry_rmsd(subunits_for_rmsd)
print(f"\n  Symmetry RMSD (first 4 subunits): {rmsd:.4f} Å")

# Packing density
pd = packing_density(assembly_coords)
print(f"  Packing density: {pd:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  OUTPUT FILES")
print("=" * 70)
for f in sorted(OUTPUT_DIR.iterdir()):
    size = f.stat().st_size
    if size > 1024:
        size_str = f"{size/1024:.1f} KB"
    else:
        size_str = f"{size} B"
    print(f"  {f.name:<40s} {size_str:>10s}")

print("\n" + "=" * 70)
print("  DESIGN COMPLETE")
print("=" * 70)
print(f"  Cage: {spec.name}")
print(f"  Symmetry: {spec.symmetry_group} ({sg.order} subunits)")
print(f"  Cavity: {spec.cavity.target_diameter_nm} nm")
print(f"  Phase: {spec.cavity.crystal_phase}")
print(f"  Best sequence: {best_name} (score={design_scores[best_idx]:.3f})")
print(f"  Files: {len(list(OUTPUT_DIR.iterdir()))} output files in {OUTPUT_DIR}/")
print(f"\n  Open {html_path} in a browser for interactive 3D visualization!")
print("=" * 70)
