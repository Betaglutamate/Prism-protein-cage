#!/usr/bin/env python3
"""
Generate enhanced PRISM cage viewer HTML.

Shows all 4 cage designs with MPNN variety, plus Design 0 glue-binder
overlay (BindCraft Relaxed + LowConfidence trajectories).  The binder is
merged into the full 12-subunit cage by renaming its chain so it appears
in the correct spatial context.

Run from the prism/ directory:
    python notebooks/generate_enhanced_viewer.py
"""

import json
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
OUTPUT = BASE / "output" / "designed_cages"
VIEWER_OUT = OUTPUT / "cage_design_viewer.html"

CAGE_PDBS = [
    OUTPUT / f"design_{i}" / "mpnn" / "packed" / f"cage_assembly_{i}_packed_{v}_1.pdb"
    for i in range(4)
    for v in range(1, 5)
]

RELAXED_DIR = OUTPUT / "design_0" / "bindcraft" / "designs" / "Trajectory" / "Relaxed"
LOWCONF_DIR = OUTPUT / "design_0" / "bindcraft" / "designs" / "Trajectory" / "LowConfidence"

# ── Design metrics (from pipeline_results.json) ────────────────────────
def load_design_info():
    with open(OUTPUT / "pipeline_results.json") as f:
        data = json.load(f)
    return data["designs"]

# ── Binder metadata ────────────────────────────────────────────────────
BINDER_METRICS = {
    "cage0_glue_l78_s706912": {
        "length": 78, "plddt": 0.83, "iptm": 0.15, "dg": -25.3,
        "sequence": "GTHTIEWEAFGMTLTYTFEPDEAGVMMVTMTVGETTLFTIAWQVYRAMMQFLMDTFPGWREFFGPHMPEMDAVHDDMM",
        "n_iface": 9, "shape_comp": 0.70, "dSASA": 712, "category": "Relaxed",
        "rmsd_to_target": 2.66,
    },
    "cage0_glue_l64_s708990": {
        "length": 64, "plddt": 0.80, "iptm": 0.16, "dg": -44.6,
        "sequence": "SGQKEWFKMVKDWVDDMIGMFEKWQKHFASMGGNNTSKEVAKMVQVAIDGLKDLRSFLESQISG",
        "n_iface": 17, "shape_comp": 0.75, "dSASA": 1276, "category": "Relaxed",
        "rmsd_to_target": 1.33,
    },
    "cage0_glue_l56_s204482": {
        "length": 56, "plddt": 0.78, "iptm": 0.16, "dg": -24.1,
        "sequence": "MAPPDMTGMTPVQPVKEWKKWTMKVADQNGWPKSWKKMVENVFDQAYDNTKQMTGM",
        "n_iface": 13, "shape_comp": 0.63, "dSASA": 1007, "category": "Relaxed",
        "rmsd_to_target": 0.80,
    },
}

# Low-confidence binders — extract length from filename, mark metrics unknown
LOWCONF_NAMES = [
    "cage0_glue_l53_s61545",
    "cage0_glue_l54_s426457",
    "cage0_glue_l57_s497434",
    "cage0_glue_l58_s717142",
    "cage0_glue_l66_s194391",
    "cage0_glue_l70_s495959",
    "cage0_glue_l77_s504296",
    "cage0_glue_l77_s997363",
]
for _n in LOWCONF_NAMES:
    _l = int(re.search(r"_l(\d+)_", _n).group(1))
    BINDER_METRICS[_n] = {
        "length": _l, "plddt": None, "iptm": None, "dg": None,
        "sequence": None, "n_iface": None, "shape_comp": None,
        "dSASA": None, "category": "LowConfidence", "rmsd_to_target": None,
    }

# ── Chain colours ──────────────────────────────────────────────────────
CHAIN_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    "#BB8FCE", "#82E0AA", "#F0B27A", "#85C1E9",
]
BINDER_COLOR = "#FFD700"   # gold
N_SUBUNITS   = 12

# ── PDB utilities ──────────────────────────────────────────────────────
def read_pdb(path: Path) -> str:
    return path.read_text()


def merge_cage_plus_binder(cage_pdb: Path, binder_pdb: Path, binder_new_chain="M") -> str:
    """
    Combine the 12-chain cage with the binder chain from a BindCraft output.
    Binder chain B is renamed to `binder_new_chain` so it doesn't collide.
    The target chain A in the binder PDB is already present in the cage.
    """
    cage_lines = cage_pdb.read_text().splitlines()
    binder_lines = binder_pdb.read_text().splitlines()

    cage_atoms = [l for l in cage_lines if l.startswith(("ATOM", "HETATM", "TER"))]

    # Keep only chain B (binder) from the binder PDB, rename to binder_new_chain
    binder_atoms = []
    for line in binder_lines:
        if line.startswith(("ATOM", "HETATM")):
            chain = line[21]
            if chain == "B":
                line = line[:21] + binder_new_chain + line[22:]
                binder_atoms.append(line)
        elif line.startswith("TER") and len(line) > 21 and line[21] == "B":
            binder_atoms.append(line[:21] + binder_new_chain + (line[22:] if len(line) > 22 else ""))

    merged = cage_atoms + ["TER"] + binder_atoms + ["END"]
    return "\n".join(merged)


def merge_binder_with_pair(pair_pdb: Path, binder_pdb: Path, binder_new_chain="M") -> str:
    """
    Combine a 2-subunit interface pair (chains A, C from subunit_pair_AC.pdb)
    with only the binder chain (chain B) from a BindCraft output PDB.
    The binder chain is renamed to `binder_new_chain` (default M).
    This gives a compact ~3-chain PDB ideal for showing the interface context.
    """
    pair_lines = pair_pdb.read_text().splitlines()
    binder_lines = binder_pdb.read_text().splitlines()

    pair_atoms = [l for l in pair_lines if l.startswith(("ATOM", "HETATM", "TER"))]

    binder_atoms = []
    for line in binder_lines:
        if line.startswith(("ATOM", "HETATM")):
            chain = line[21]
            if chain == "B":
                line = line[:21] + binder_new_chain + line[22:]
                binder_atoms.append(line)
        elif line.startswith("TER") and len(line) > 21 and line[21] == "B":
            binder_atoms.append(line[:21] + binder_new_chain + (line[22:] if len(line) > 22 else ""))

    merged = pair_atoms + ["TER"] + binder_atoms + ["END"]
    return "\n".join(merged)


def escape_pdb(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


# ── HTML helpers ──────────────────────────────────────────────────────
def metric_value(val, fmt=".2f", good_thresh=None, bad_thresh=None, invert=False):
    if val is None:
        return '<span class="na">N/A</span>'
    fv = f"{val:{fmt}}"
    if good_thresh is not None and bad_thresh is not None:
        if invert:
            cls = "good" if val <= good_thresh else ("bad" if val >= bad_thresh else "warn")
        else:
            cls = "good" if val >= good_thresh else ("bad" if val <= bad_thresh else "warn")
        return f'<span class="{cls}">{fv}</span>'
    return f"<span>{fv}</span>"


# ── Main generation ────────────────────────────────────────────────────
def generate():
    designs = load_design_info()

    # ── Gather binder PDB data ──────────────────────────────────────────
    # Use the 2-subunit pair (AC) as binder context — much smaller than full cage
    # while still showing the interface where the binder was designed to bind.
    subunit_pair = OUTPUT / "design_0" / "bindcraft" / "subunit_pair_AC.pdb"

    all_binders = {}   # name -> {"merged_pdb": str, "meta": dict}
    for name, meta in BINDER_METRICS.items():
        folder = RELAXED_DIR if meta["category"] == "Relaxed" else LOWCONF_DIR
        binder_path = folder / f"{name}.pdb"
        if binder_path.exists() and subunit_pair.exists():
            # Merge: subunit_pair (chains A,C) + binder chain B renamed to M
            merged = merge_binder_with_pair(subunit_pair, binder_path, "M")
            all_binders[name] = {"merged_pdb": merged, "meta": meta}
        else:
            print(f"  [skip] {name} — file missing")

    # ── For each design, collect 2 packed variant PDBs (v1 + v3 for variety) ──
    design_variants = {}  # idx -> list of (label, pdb_text)
    for d in designs:
        idx = d["idx"]
        variants = []
        for v in [1, 3]:  # Show 2 variants for diversity, keep file size reasonable
            p = OUTPUT / f"design_{idx}" / "mpnn" / "packed" / f"cage_assembly_{idx}_packed_{v}_1.pdb"
            if p.exists():
                variants.append((f"MPNN v{v}", p.read_text()))
        design_variants[idx] = variants

    # ── Build HTML ──────────────────────────────────────────────────────
    html = build_html(designs, design_variants, all_binders)
    VIEWER_OUT.write_text(html)
    size_kb = VIEWER_OUT.stat().st_size / 1024
    print(f"✅  Written {VIEWER_OUT}  ({size_kb:.0f} KB)")


def build_html(designs, design_variants, all_binders):
    parts = [HTML_HEAD]

    # ── Pipeline description ────────────────────────────────────────────
    parts.append("""
<div class="hero">
  <h1>&#x1F9EC; PRISM Protein Cage Designer</h1>
  <p class="subheader">RFdiffusion2 backbone &rarr; LigandMPNN homo-oligomeric sequence &rarr; BindCraft interface optimisation</p>
</div>

<div class="bio-note">
  <div class="bio-grid">
    <div>
      <strong>&#x1F501; Design pipeline</strong><br>
      <ol style="margin:6px 0 0 16px;padding:0;line-height:1.8">
        <li><strong>RFdiffusion2</strong> &mdash; SE(3) flow matching generates 200-residue subunit backbones</li>
        <li><strong>PRISM symmetry engine</strong> &mdash; 12-subunit T-cage at 50&nbsp;&Aring; radius</li>
        <li><strong>LigandMPNN</strong> &mdash; homo-oligomeric sequence design (identical chains)</li>
        <li><strong>BindCraft</strong> &mdash; AF2-hallucination glue-binder optimisation at A&ndash;C interface</li>
      </ol>
    </div>
    <div>
      <strong>&#x1F4CA; Run summary</strong><br>
      <table class="summary-table" style="margin-top:8px">
        <tr><td>Cage designs</td><td class="num">4</td></tr>
        <tr><td>Subunits / cage</td><td class="num">12 (T symmetry)</td></tr>
        <tr><td>Residues / subunit</td><td class="num">200</td></tr>
        <tr><td>Glue binders generated</td><td class="num">11 (Design 0)</td></tr>
        <tr><td>Relaxed binders</td><td class="num">3</td></tr>
        <tr><td>Best &Delta;G</td><td class="num good">&minus;44.6 REU</td></tr>
      </table>
    </div>
  </div>
</div>
""")

    # ── 4-design cage grid ──────────────────────────────────────────────
    parts.append('<h2 class="section-title">&#x1F3D7; Cage Assemblies</h2>')
    parts.append('<div class="tab-selector cage-selector">')
    for i, d in enumerate(designs):
        active = "active" if i == 0 else ""
        parts.append(f'<button class="sel-tab {active}" onclick="showDesign({i})" id="cage-tab-{i}">Design {i}</button>')
    parts.append('</div>')

    for i, d in enumerate(designs):
        m = d.get("cage_metrics", {})
        mpnn = d.get("mpnn_metrics", {})
        seq = d.get("best_sequence", "")
        visible = "block" if i == 0 else "none"
        conf = mpnn.get("overall_confidence", 0)
        conf_cls = "good" if conf > 0.5 else "warn" if conf > 0.3 else "bad"
        n_binders = len(all_binders) if i == 0 else 0
        variants = design_variants.get(i, [])

        parts.append(f'<div class="cage-card" id="cage-card-{i}" style="display:{visible}">')
        parts.append(f'  <div class="cage-layout">')
        parts.append(f'    <div class="cage-viewer-col">')

        # variant sub-tabs
        if len(variants) > 1:
            parts.append(f'      <div class="variant-tabs">')
            for vi, (vlabel, _vpdb) in enumerate(variants):
                cls = "active" if vi == 0 else ""
                parts.append(f'        <button class="vtab {cls}" onclick="showVariant({i},{vi})" id="vtab-{i}-{vi}">{vlabel}</button>')
            parts.append(f'      </div>')

        for vi, (vlabel, _vpdb) in enumerate(variants):
            dsp = "block" if vi == 0 else "none"
            parts.append(f'      <div id="vview-{i}-{vi}" style="display:{dsp}">')
            parts.append(f'        <div id="cage_viewer_{i}_{vi}" class="mol-viewer"></div>')
            parts.append(f'      </div>')

        parts.append(f'    </div>')  # cage-viewer-col

        # metrics column
        parts.append(f'    <div class="cage-info-col">')
        parts.append(f'      <h3>Design {i} &mdash; T&#8209;cage</h3>')
        if n_binders:
            parts.append(f'      <span class="badge designed">&#x2705; {n_binders} glue binders</span>')
        else:
            parts.append(f'      <span class="badge pending">Binder pending</span>')
        parts.append(f'      <table class="metric-table">')
        parts.append(f'        <tr><th colspan="2">Cage geometry</th></tr>')
        parts.append(f'        <tr><td>Cavity &oslash;</td><td>{m.get("cavity_diameter_nm","N/A")} nm</td></tr>')
        parts.append(f'        <tr><td>R<sub>g</sub></td><td>{m.get("rog_A","N/A")} &Aring;</td></tr>')
        parts.append(f'        <tr><td>Shell</td><td>{m.get("inner_radius_A","?")}–{m.get("outer_radius_A","?")} &Aring;</td></tr>')
        parts.append(f'        <tr><td>Close interfaces</td><td>{m.get("n_close_interfaces","N/A")}</td></tr>')
        parts.append(f'        <tr><td>Clashes (2&nbsp;&Aring;)</td><td>{m.get("clashes_2A",0)}</td></tr>')
        parts.append(f'        <tr><th colspan="2">LigandMPNN sequence</th></tr>')
        parts.append(f'        <tr><td>Confidence</td><td class="{conf_cls}">{conf:.3f}</td></tr>')
        parts.append(f'        <tr><td>Seq recovery</td><td>{mpnn.get("seq_rec","N/A")}</td></tr>')
        parts.append(f'        <tr><td>Interface res</td><td>{d.get("n_interface_residues","N/A")}</td></tr>')
        parts.append(f'      </table>')
        seq_disp = seq[:160] + ("…" if len(seq) > 160 else "") if seq else "N/A"
        parts.append(f'      <div class="seq-label">Chain A sequence (200 res):</div>')
        parts.append(f'      <div class="seq-box">{seq_disp}</div>')
        parts.append(f'    </div>')  # cage-info-col
        parts.append(f'  </div>')  # cage-layout
        parts.append(f'</div>')  # cage-card

    # ── Binder section (Design 0) ──────────────────────────────────────
    parts.append('<h2 class="section-title">&#x1F9F2; Glue Binder Analysis &mdash; Design 0 (A&ndash;C Interface)</h2>')
    parts.append("""
<div class="bio-note" style="margin-bottom:12px">
  <strong>Interface targeting</strong>: Hotspot residues 72&ndash;89, 124, 127 on the A&ndash;C subunit interface (contact distance &lt;2.1&nbsp;&Aring;).
  BindCraft ran AF2 hallucination to design a peptide that bridges two adjacent cage subunits.
  <em>iPTM &lt; 0.5 indicates early-stage designs — these are initial trajectories; &Delta;G and shape complementarity are the primary quality filters.</em><br>
  Viewer shows the two target subunits (A = <span style="color:#FF6B6B">&#x25A0;</span> red, C = <span style="color:#45B7D1">&#x25A0;</span> blue) and the designed glue binder (<span style="color:#FFD700">&#x25A0;</span> gold).
</div>
""")

    if all_binders:
        # Tab bar
        parts.append('<div class="tab-selector binder-selector">')
        for bi, (bname, bdata) in enumerate(all_binders.items()):
            active = "active" if bi == 0 else ""
            meta = bdata["meta"]
            cat_cls = "rel-tab" if meta["category"] == "Relaxed" else "lc-tab"
            label = f"L{meta['length']}"
            parts.append(
                f'<button class="sel-tab {active} {cat_cls}" '
                f'onclick="showBinder({bi})" id="btab-{bi}" '
                f'title="{bname}">{label}</button>'
            )
        parts.append('</div>')
        parts.append('<div style="font-size:11px;color:#8b949e;margin:-6px 0 10px 4px">'
                     '&#x1F7E1; Relaxed &nbsp; &#x26AA; LowConfidence &nbsp;·&nbsp; '
                     'Label = binder length</div>')

        for bi, (bname, bdata) in enumerate(all_binders.items()):
            meta = bdata["meta"]
            visible = "block" if bi == 0 else "none"
            parts.append(f'<div class="binder-card" id="binder-card-{bi}" style="display:{visible}">')
            parts.append(f'  <div class="cage-layout">')
            parts.append(f'    <div class="cage-viewer-col">')
            parts.append(f'      <div id="binder_viewer_{bi}" class="mol-viewer tall-viewer"></div>')
            parts.append(f'    </div>')

            # metrics
            plddt_str = f'{meta["plddt"]:.2f}' if meta["plddt"] else "N/A"
            iptm_str  = f'{meta["iptm"]:.2f}'  if meta["iptm"]  else "N/A"
            dg_str    = f'{meta["dg"]:.1f}'    if meta["dg"]    else "N/A"
            sc_str    = f'{meta["shape_comp"]:.2f}' if meta["shape_comp"] else "N/A"
            dsasa_str = f'{meta["dSASA"]:.0f}' if meta["dSASA"] else "N/A"
            rmsd_str  = f'{meta["rmsd_to_target"]:.2f}' if meta["rmsd_to_target"] else "N/A"
            seq_str   = meta["sequence"] or "N/A"

            plddt_cls = ("good" if meta["plddt"] and meta["plddt"] >= 0.8
                         else "warn" if meta["plddt"] and meta["plddt"] >= 0.6
                         else "bad") if meta["plddt"] else "na"
            iptm_cls  = ("good" if meta["iptm"] and meta["iptm"] >= 0.45
                         else "warn" if meta["iptm"] and meta["iptm"] >= 0.25
                         else "bad") if meta["iptm"] else "na"
            dg_cls    = ("good" if meta["dg"] and meta["dg"] <= -30
                         else "warn" if meta["dg"] and meta["dg"] <= -15
                         else "bad") if meta["dg"] else "na"

            cat_badge = ("designed" if meta["category"] == "Relaxed" else "pending")
            parts.append(f'    <div class="cage-info-col">')
            parts.append(f'      <h3>{bname.replace("cage0_glue_","")}</h3>')
            parts.append(f'      <span class="badge {cat_badge}">{meta["category"]}</span>')
            parts.append(f'      <table class="metric-table">')
            parts.append(f'        <tr><th colspan="2">AF2 prediction quality</th></tr>')
            parts.append(f'        <tr><td>pLDDT</td><td class="{plddt_cls}">{plddt_str}</td></tr>')
            parts.append(f'        <tr><td>iPTM</td><td class="{iptm_cls}">{iptm_str}</td></tr>')
            parts.append(f'        <tr><th colspan="2">Rosetta interface energy</th></tr>')
            parts.append(f'        <tr><td>&Delta;G (REU)</td><td class="{dg_cls}">{dg_str}</td></tr>')
            parts.append(f'        <tr><td>dSASA (&Aring;&sup2;)</td><td>{dsasa_str}</td></tr>')
            parts.append(f'        <tr><td>Shape compl.</td><td>{sc_str}</td></tr>')
            parts.append(f'        <tr><th colspan="2">Structure</th></tr>')
            parts.append(f'        <tr><td>Length (res)</td><td>{meta["length"]}</td></tr>')
            parts.append(f'        <tr><td>Interface res</td><td>{meta["n_iface"] or "N/A"}</td></tr>')
            parts.append(f'        <tr><td>Target RMSD</td><td>{rmsd_str}</td></tr>')
            parts.append(f'      </table>')
            if seq_str != "N/A":
                parts.append(f'      <div class="seq-label">Binder sequence:</div>')
                parts.append(f'      <div class="seq-box">{seq_str}</div>')
            parts.append(f'    </div>')  # info col
            parts.append(f'  </div>')  # layout
            parts.append(f'</div>')  # binder-card
    else:
        parts.append('<p style="color:#8b949e;padding:20px">No binder PDB files found.</p>')

    # ── Footer ─────────────────────────────────────────────────────────
    parts.append("""
<div class="footer">
  PRISM cage design toolkit &mdash; RFdiffusion2 &rarr; LigandMPNN &rarr; BindCraft<br>
  <a href="https://github.com/Betaglutamate/Prism-protein-cage" target="_blank" style="color:#58a6ff">
    github.com/Betaglutamate/Prism-protein-cage</a>
</div>
</body></html>
""")

    # ── JavaScript ─────────────────────────────────────────────────────
    js_parts = ["<script>"]
    js_parts.append("""
function showDesign(idx) {
  document.querySelectorAll('.cage-card').forEach(function(el){el.style.display='none';});
  document.querySelectorAll('.cage-selector .sel-tab').forEach(function(b){b.classList.remove('active');});
  document.getElementById('cage-card-'+idx).style.display='block';
  document.getElementById('cage-tab-'+idx).classList.add('active');
}
function showVariant(design, vi) {
  for(var v=0;v<2;v++){
    var el=document.getElementById('vview-'+design+'-'+v);
    var tb=document.getElementById('vtab-'+design+'-'+v);
    if(el){el.style.display=(v===vi?'block':'none');}
    if(tb){tb.classList.toggle('active',v===vi);}
  }
}
function showBinder(idx) {
  document.querySelectorAll('.binder-card').forEach(function(el){el.style.display='none';});
  document.querySelectorAll('.binder-selector .sel-tab').forEach(function(b){b.classList.remove('active');});
  document.getElementById('binder-card-'+idx).style.display='block';
  document.getElementById('btab-'+idx).classList.add('active');
}
""")

    # Cage viewers
    js_parts.append("document.addEventListener('DOMContentLoaded', function() {")
    for i, d in enumerate(designs):
        variants = design_variants.get(i, [])
        for vi, (_vlabel, vpdb_text) in enumerate(variants):
            pdb_esc = escape_pdb(vpdb_text)
            js_parts.append(f"  (function() {{")
            js_parts.append(f"    var el = document.getElementById('cage_viewer_{i}_{vi}');")
            js_parts.append(f"    if (!el) return;")
            js_parts.append(f"    var viewer = $3Dmol.createViewer(el, {{backgroundColor: '#0d1117'}});")
            js_parts.append(f"    viewer.addModel('{pdb_esc}', 'pdb');")
            for ci in range(N_SUBUNITS):
                chain = chr(65 + ci)
                color = CHAIN_COLORS[ci % len(CHAIN_COLORS)]
                js_parts.append(
                    f"    viewer.setStyle({{chain: '{chain}'}}, "
                    f"{{cartoon: {{color: '{color}', opacity: 0.85}}}});")
            js_parts.append(f"    viewer.zoomTo(); viewer.render(); viewer.spin('y', 0.4);")
            js_parts.append(f"  }})();")

    # Binder viewers — show 2-subunit pair (A, C) + binder (M)
    PAIR_CHAIN_COLORS = {"A": CHAIN_COLORS[0], "C": CHAIN_COLORS[2]}
    for bi, (bname, bdata) in enumerate(all_binders.items()):
        merged_pdb = bdata["merged_pdb"]
        pdb_esc = escape_pdb(merged_pdb)
        js_parts.append(f"  (function() {{")
        js_parts.append(f"    var el = document.getElementById('binder_viewer_{bi}');")
        js_parts.append(f"    if (!el) return;")
        js_parts.append(f"    var viewer = $3Dmol.createViewer(el, {{backgroundColor: '#0d1117'}});")
        js_parts.append(f"    viewer.addModel('{pdb_esc}', 'pdb');")
        js_parts.append(f"    viewer.setStyle({{chain: 'A'}}, {{cartoon: {{color: '{CHAIN_COLORS[0]}', opacity: 0.75}}}});")
        js_parts.append(f"    viewer.setStyle({{chain: 'C'}}, {{cartoon: {{color: '{CHAIN_COLORS[2]}', opacity: 0.75}}}});")
        # Binder chain M in gold, full opacity
        js_parts.append(
            f"    viewer.setStyle({{chain: 'M'}}, "
            f"{{cartoon: {{color: '{BINDER_COLOR}', opacity: 1.0, thickness: 0.6}}}});")
        js_parts.append(f"    viewer.zoomTo(); viewer.render(); viewer.spin('y', 0.4);")
        js_parts.append(f"  }})();")

    js_parts.append("});")
    js_parts.append("</script>")

    full_html = "\n".join(parts) + "\n" + "\n".join(js_parts)
    return full_html


HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PRISM Cage Designer — Cage &amp; Glue Binder Viewer</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #0d1117;
  color: #c9d1d9;
  margin: 0;
  padding: 0 20px 40px;
}

/* ── Hero ─────────────────────────────────────────────────────────── */
.hero { text-align: center; padding: 32px 0 16px; }
.hero h1 { color: #58a6ff; font-size: 2rem; margin: 0 0 8px; }
.subheader { color: #8b949e; font-size: 14px; margin: 0; }

/* ── Bio-note card ────────────────────────────────────────────────── */
.bio-note {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 16px 20px;
  max-width: 1300px;
  margin: 0 auto 20px;
  font-size: 13px;
  line-height: 1.7;
}
.bio-note strong { color: #58a6ff; }
.bio-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
@media (max-width: 700px) { .bio-grid { grid-template-columns: 1fr; } }

.summary-table td { padding: 2px 12px 2px 0; }
.summary-table td.num { font-weight: bold; color: #e6edf3; }

/* ── Section titles ──────────────────────────────────────────────── */
.section-title {
  color: #79c0ff;
  font-size: 1.1rem;
  border-bottom: 1px solid #21262d;
  padding-bottom: 6px;
  max-width: 1300px;
  margin: 32px auto 14px;
}

/* ── Tab selectors ───────────────────────────────────────────────── */
.tab-selector {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  max-width: 1300px;
  margin: 0 auto 14px;
}
.sel-tab {
  padding: 6px 18px;
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 20px;
  color: #8b949e;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}
.sel-tab:hover { background: #2d333b; color: #c9d1d9; }
.sel-tab.active { background: #1f6feb; border-color: #388bfd; color: white; font-weight: 600; }
.sel-tab.rel-tab { border-left: 3px solid #f1c40f; }
.sel-tab.lc-tab  { border-left: 3px solid #444; }

/* ── Cards ───────────────────────────────────────────────────────── */
.cage-card, .binder-card {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 12px;
  padding: 16px 20px;
  max-width: 1300px;
  margin: 0 auto 16px;
}
.cage-layout {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 20px;
}
@media (max-width: 900px) { .cage-layout { grid-template-columns: 1fr; } }

/* ── Variant sub-tabs ────────────────────────────────────────────── */
.variant-tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 8px;
}
.vtab {
  padding: 3px 12px;
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 6px 6px 0 0;
  color: #8b949e;
  cursor: pointer;
  font-size: 12px;
}
.vtab.active { background: #0d1117; color: #58a6ff; border-bottom-color: #0d1117; }

/* ── 3Dmol viewers ───────────────────────────────────────────────── */
.mol-viewer {
  width: 100%;
  height: 450px;
  border-radius: 8px;
  overflow: hidden;
  background: #0d1117;
}
.tall-viewer { height: 500px; }

/* ── Info column ─────────────────────────────────────────────────── */
.cage-info-col h3 { margin: 0 0 8px; color: #e6edf3; font-size: 1rem; }

.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  margin-bottom: 10px;
}
.badge.designed { background: #238636; color: #fff; }
.badge.pending  { background: #373e47; color: #8b949e; }

.metric-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin-bottom: 8px;
}
.metric-table th {
  text-align: left;
  color: #58a6ff;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 6px 4px 2px;
  border-bottom: 1px solid #21262d;
}
.metric-table td {
  padding: 3px 4px;
  border-bottom: 1px solid #1c2128;
}
.metric-table td:last-child { text-align: right; font-weight: 600; }

/* ── Sequence box ────────────────────────────────────────────────── */
.seq-label { color: #8b949e; font-size: 11px; margin: 8px 0 2px; }
.seq-box {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 4px;
  padding: 6px 8px;
  font-family: 'Courier New', monospace;
  font-size: 11px;
  word-break: break-all;
  line-height: 1.5;
  max-height: 70px;
  overflow-y: auto;
  color: #b1bac4;
}

/* ── Colour classes ──────────────────────────────────────────────── */
.good { color: #3fb950 !important; }
.warn { color: #d29922 !important; }
.bad  { color: #f85149 !important; }
.na   { color: #8b949e !important; font-style: italic; }

/* ── Footer ──────────────────────────────────────────────────────── */
.footer {
  text-align: center;
  color: #8b949e;
  font-size: 12px;
  margin-top: 40px;
  padding-top: 12px;
  border-top: 1px solid #21262d;
  max-width: 1300px;
  margin-left: auto;
  margin-right: auto;
}
</style>
</head>
<body>
"""

if __name__ == "__main__":
    generate()
