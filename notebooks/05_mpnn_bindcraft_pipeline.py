#!/usr/bin/env python3
"""
Sequence Design & Interface Optimization Pipeline
===================================================
Step 1: LigandMPNN — assign sequences to all encapsulin-inspired cage backbones
Step 2: BindCraft  — optimize inter-subunit binding interfaces via AF2 hallucination
Step 3: Scoring    — rank designs by MPNN confidence + BindCraft metrics
Step 4: Viewer     — interactive HTML with per-chain coloring & designed sequences
"""

import sys, os, json, subprocess, shutil, time
import numpy as np
from pathlib import Path
from glob import glob

# ── Configuration ─────────────────────────────────────────────────────
PRISM_ROOT    = Path(__file__).resolve().parent.parent
CAGE_DIR      = PRISM_ROOT / "output" / "encapsulin_cage_results"
OUTPUT_DIR    = PRISM_ROOT / "output" / "designed_cages"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LigandMPNN paths
RFD2_ROOT     = Path("/home/markzurbruegg/library-design/rfdiffusion2-lib/RFdiffusion2")
MPNN_DIR      = RFD2_ROOT / "fused_mpnn"
MPNN_WEIGHTS  = RFD2_ROOT / "rf_diffusion/third_party_model_weights/ligand_mpnn/s25_r010_t300_p.pt"
SC_WEIGHTS    = RFD2_ROOT / "rf_diffusion/third_party_model_weights/ligand_mpnn/s_300756.pt"

# BindCraft paths
BINDCRAFT_DIR = Path("/home/markzurbruegg/library-design/BindCraft")
BINDCRAFT_ENV = Path("/home/markzurbruegg/miniconda3/envs/BindCraft")

# Design parameters
N_MPNN_SEQS      = 4       # sequences per cage backbone
MPNN_TEMPERATURE = 0.1     # low T → higher confidence
MPNN_BATCH_SIZE  = 2
N_SUBUNITS       = 12

# BindCraft parameters
BINDER_LENGTH    = (40, 70)     # residues for interface binder
N_BINDER_DESIGNS = 10           # designs per cage
INTERFACE_CONTACT_CUTOFF = 8.0  # Å — residues within this distance are "interface"

# ── Helper functions ──────────────────────────────────────────────────

def run_mpnn(cage_pdb: Path, out_dir: Path, n_seqs: int = N_MPNN_SEQS,
             temperature: float = MPNN_TEMPERATURE,
             pack_side_chains: bool = True) -> dict:
    """Run LigandMPNN on a cage assembly PDB with homo-oligomer mode.
    
    Returns dict with keys: seqs_fasta, backbone_pdbs, packed_pdbs, metrics
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_batches = max(1, n_seqs // MPNN_BATCH_SIZE)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{RFD2_ROOT}:{MPNN_DIR}:{env.get('PYTHONPATH', '')}"
    
    cmd = [
        sys.executable, str(MPNN_DIR / "run.py"),
        "--model_type", "ligand_mpnn",
        "--checkpoint_ligand_mpnn", str(MPNN_WEIGHTS),
        "--checkpoint_path_sc", str(SC_WEIGHTS),
        "--pdb_path", str(cage_pdb),
        "--out_folder", str(out_dir),
        "--homo_oligomer", "1",
        "--batch_size", str(MPNN_BATCH_SIZE),
        "--number_of_batches", str(n_batches),
        "--temperature", str(temperature),
        "--seed", "42",
        "--pack_side_chains", "1" if pack_side_chains else "0",
        "--number_of_packs_per_design", "1",
        "--verbose", "1",
    ]
    
    log_path = out_dir / "mpnn.log"
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, env=env, cwd=str(MPNN_DIR),
                                stdout=log, stderr=subprocess.STDOUT,
                                timeout=600)
    
    if result.returncode != 0:
        log_text = log_path.read_text()[-1000:]
        print(f"    [WARN] MPNN exited with code {result.returncode}")
        print(f"    Log tail: {log_text[-500:]}")
    
    # Collect outputs
    seqs_fastas = sorted(out_dir.glob("seqs/*.fa"))
    backbone_pdbs = sorted(out_dir.glob("backbones/*.pdb"))
    packed_pdbs = sorted(out_dir.glob("packed/*.pdb"))
    
    # Parse sequences and metrics from FASTA
    sequences = []
    metrics_list = []
    for fa in seqs_fastas:
        with open(fa) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">") and "id=" in line:
                    # Parse header for metrics
                    meta = {}
                    for part in line.split(", "):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            k = k.strip().lstrip(">").strip()
                            try:
                                meta[k] = float(v)
                            except ValueError:
                                meta[k] = v
                    metrics_list.append(meta)
                elif line and not line.startswith(">"):
                    # Sequence line — take just chain A (first segment before /)
                    chain_a_seq = line.split("/")[0]
                    sequences.append(chain_a_seq)
    
    return {
        "seqs_fastas": seqs_fastas,
        "backbone_pdbs": backbone_pdbs,
        "packed_pdbs": packed_pdbs,
        "sequences": sequences,
        "metrics": metrics_list,
        "log_path": log_path,
    }


def find_interface_residues(cage_pdb: Path, cutoff: float = INTERFACE_CONTACT_CUTOFF) -> list[int]:
    """Find residues at inter-subunit interfaces (chain A residues close to other chains).
    
    Returns list of residue numbers on chain A that are within `cutoff` Å of any
    atom on adjacent chains.
    """
    chain_atoms = {}  # chain -> list of [x,y,z]
    chain_a_res = {}  # resnum -> list of [x,y,z]
    
    with open(cage_pdb) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            resnum = int(line[22:26])
            
            chain_atoms.setdefault(chain, []).append([x, y, z])
            if chain == "A":
                chain_a_res.setdefault(resnum, []).append([x, y, z])
    
    if "A" not in chain_atoms or len(chain_atoms) < 2:
        return list(range(1, 51))  # fallback
    
    # Get atoms from all non-A chains
    other_coords = []
    for ch, coords in chain_atoms.items():
        if ch != "A":
            other_coords.extend(coords)
    other_coords = np.array(other_coords)
    
    # Find chain A residues near interface
    interface_residues = []
    for resnum, res_coords in sorted(chain_a_res.items()):
        res_coords_arr = np.array(res_coords)
        # Minimum distance from any atom of this residue to any other-chain atom
        # Use subsampled other_coords for speed
        step = max(1, len(other_coords) // 2000)
        dists = np.linalg.norm(
            res_coords_arr[:, None, :] - other_coords[None, ::step, :], axis=-1
        )
        min_dist = dists.min()
        if min_dist < cutoff:
            interface_residues.append(resnum)
    
    return interface_residues


def extract_subunit_pair(cage_pdb: Path, output_pdb: Path,
                         chains: tuple[str, str] = ("A", "B")) -> Path:
    """Extract a two-chain subunit pair from the cage assembly for BindCraft."""
    with open(cage_pdb) as fin, open(output_pdb, "w") as fout:
        fout.write("REMARK   Subunit pair for BindCraft interface optimization\n")
        for line in fin:
            if line.startswith("ATOM") or line.startswith("TER"):
                chain = line[21]
                if chain in chains:
                    fout.write(line)
        fout.write("END\n")
    return output_pdb


def run_bindcraft(target_pdb: Path, interface_residues: list[int],
                  output_dir: Path, target_chain: str = "A") -> dict:
    """Run BindCraft to optimize the binding interface.
    
    Returns dict with design results and metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if BindCraft env exists
    bindcraft_python = BINDCRAFT_ENV / "bin" / "python"
    if not bindcraft_python.exists():
        return {"status": "skipped", "reason": "BindCraft env not installed yet"}
    
    # Create target settings
    hotspot_str = ",".join(str(r) for r in interface_residues[:20])  # limit to top 20
    
    target_settings = {
        "target_pdb": str(target_pdb),
        "target_chain": target_chain,
        "target_hotspot_residues": hotspot_str,
        "binder_length": f"{BINDER_LENGTH[0]}-{BINDER_LENGTH[1]}",
    }
    
    settings_path = output_dir / "target_settings.json"
    with open(settings_path, "w") as f:
        json.dump(target_settings, f, indent=2)
    
    # Create filter settings
    filter_settings = {
        "pae_interaction_cutoff": 10.0,
        "plddt_cutoff": 70.0,
        "iptm_cutoff": 0.5,
        "clash_cutoff": 1.0,
    }
    filter_path = output_dir / "filter_settings.json"
    with open(filter_path, "w") as f:
        json.dump(filter_settings, f, indent=2)
    
    # Create advanced settings
    advanced_settings = {
        "num_recycles": 3,
        "design_iterations": 50,
        "soft_iterations": 25,
        "hard_iterations": 5,
        "use_multimer": True,
    }
    advanced_path = output_dir / "advanced_settings.json"
    with open(advanced_path, "w") as f:
        json.dump(advanced_settings, f, indent=2)
    
    # Build command
    cmd = [
        str(bindcraft_python),
        str(BINDCRAFT_DIR / "bindcraft.py"),
        str(settings_path),
        str(filter_path),
        str(advanced_path),
        "--num_designs", str(N_BINDER_DESIGNS),
        "--output_dir", str(output_dir),
    ]
    
    log_path = output_dir / "bindcraft.log"
    print(f"    Running BindCraft ({N_BINDER_DESIGNS} designs)...")
    
    try:
        with open(log_path, "w") as log:
            result = subprocess.run(cmd, cwd=str(BINDCRAFT_DIR),
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=3600)  # 1 hour timeout
        
        if result.returncode != 0:
            log_text = log_path.read_text()[-500:]
            return {"status": "failed", "returncode": result.returncode,
                    "log_tail": log_text}
        
        # Parse results
        design_pdbs = sorted(output_dir.glob("*.pdb"))
        metrics = []
        
        # Try to find metrics CSV
        for csv_path in output_dir.glob("*.csv"):
            import csv
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
        
        return {
            "status": "success",
            "n_designs": len(design_pdbs),
            "design_pdbs": [str(p) for p in design_pdbs],
            "metrics": metrics,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def generate_final_viewer(designs: list[dict], output_path: Path):
    """Generate interactive HTML viewer showing ALL designed cages with sequences."""
    
    chain_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
        "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
        "#BB8FCE", "#82E0AA", "#F0B27A", "#85C1E9",
    ]
    
    html = ["""<!DOCTYPE html>
<html><head>
<title>PRISM Cage Design Pipeline — MPNN + BindCraft Results</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9;
       margin: 0; padding: 20px; }
h1 { color: #58a6ff; text-align: center; margin-bottom: 4px; }
h2 { color: #79c0ff; border-bottom: 1px solid #21262d; padding-bottom: 8px;
     font-size: 15px; margin-bottom: 8px; }
.subheader { text-align: center; color: #8b949e; margin-bottom: 20px; font-size: 14px; }
.grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;
        max-width: 1400px; margin: 0 auto; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 16px; }
.viewer-container { position: relative; width: 100%; height: 400px;
                    border-radius: 8px; overflow: hidden; }
.metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 4px;
           font-size: 13px; margin-top: 8px; }
.metric { padding: 4px 8px; background: #0d1117; border-radius: 4px; }
.metric span { color: #58a6ff; font-weight: bold; }
.seq-box { margin-top: 8px; padding: 8px; background: #0d1117; border-radius: 4px;
           font-family: 'Courier New', monospace; font-size: 11px; word-break: break-all;
           max-height: 60px; overflow-y: auto; line-height: 1.4; }
.seq-label { color: #8b949e; font-size: 11px; margin-bottom: 2px; }
.good { color: #3fb950; } .warn { color: #d29922; } .bad { color: #f85149; }
.bio-note { background: #1c2128; border: 1px solid #30363d; border-radius: 8px;
            padding: 16px; max-width: 1400px; margin: 0 auto 20px; font-size: 14px;
            line-height: 1.6; }
.bio-note strong { color: #58a6ff; }
.tab-bar { display: flex; gap: 4px; margin-bottom: 8px; }
.tab { padding: 4px 12px; background: #21262d; border: 1px solid #30363d;
       border-radius: 6px 6px 0 0; cursor: pointer; font-size: 12px; color: #8b949e; }
.tab.active { background: #0d1117; color: #58a6ff; border-bottom-color: #0d1117; }
.iface-badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
               font-size: 11px; margin-top: 4px; }
.iface-badge.designed { background: #238636; color: white; }
.iface-badge.pending { background: #30363d; color: #8b949e; }
</style>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head><body>
<h1>&#x1F9EC; Encapsulin-Inspired Cage Designs</h1>
<p class="subheader">RFdiffusion2 backbone &rarr; LigandMPNN sequence &rarr; BindCraft interface optimization</p>

<div class="bio-note">
  <strong>Design Pipeline:</strong><br>
  1. <strong>RFdiffusion2</strong> generates 200-residue subunit backbones via SE(3) flow matching<br>
  2. Subunits positioned at 50 &Aring; cage radius (mimicking encapsulin geometry)<br>
  3. <strong>Tetrahedral symmetry</strong> expands to 12-subunit cage (2,400 residues total)<br>
  4. <strong>LigandMPNN</strong> designs homo-oligomeric sequences (identical across all 12 chains)<br>
  5. <strong>BindCraft</strong> optimizes inter-subunit interfaces via AF2 hallucination
</div>

<div class="grid">
"""]
    
    for i, d in enumerate(designs):
        m = d.get("cage_metrics", {})
        mpnn = d.get("mpnn_metrics", {})
        seq = d.get("best_sequence", "")
        n_res = d.get("n_residues", 200)
        
        # Determine which PDB to show (packed if available, else backbone)
        show_pdb = d.get("display_pdb", d.get("cage_pdb", ""))
        
        # Quality classes
        conf = mpnn.get("overall_confidence", 0)
        conf_cls = "good" if conf > 0.5 else "warn" if conf > 0.3 else "bad"
        
        iface_status = d.get("bindcraft_status", "pending")
        iface_cls = "designed" if iface_status == "success" else "pending"
        iface_label = "Interface optimized" if iface_status == "success" else "Pending"
        
        html.append(f"""
<div class="card">
  <h2>Design {i} &mdash; T-cage ({n_res} res/chain, {N_SUBUNITS} chains)
    <span class="iface-badge {iface_cls}">{iface_label}</span>
  </h2>
  <div id="viewer_{i}" class="viewer-container"></div>
  <div class="metrics">
    <div class="metric">Cavity &oslash;: <span>{m.get('cavity_diameter_nm', 'N/A')} nm</span></div>
    <div class="metric">R<sub>g</sub>: <span>{m.get('rog_A', 'N/A')} &Aring;</span></div>
    <div class="metric">Shell: <span>{m.get('inner_radius_A', '?')}&ndash;{m.get('outer_radius_A', '?')} &Aring;</span></div>
    <div class="metric">MPNN conf: <span class="{conf_cls}">{conf:.3f}</span></div>
    <div class="metric">Seq recovery: <span>{mpnn.get('seq_rec', 'N/A')}</span></div>
    <div class="metric">Min contact: <span>{m.get('min_contact_dist_A', 'N/A')} &Aring;</span></div>
    <div class="metric">Interface res: <span>{d.get('n_interface_residues', 'N/A')}</span></div>
    <div class="metric">Clashes: <span>{m.get('clashes_2A', 0)}</span></div>
  </div>
  <div class="seq-label">Best MPNN sequence (chain A, {n_res} residues):</div>
  <div class="seq-box">{seq if seq else 'N/A'}</div>
</div>
""")
    
    # JavaScript for viewer initialization
    init_lines = ["\n<script>", "document.addEventListener('DOMContentLoaded', function() {"]
    
    for i, d in enumerate(designs):
        show_pdb = d.get("display_pdb", d.get("cage_pdb", ""))
        if not show_pdb or not Path(show_pdb).exists():
            continue
            
        with open(show_pdb) as f:
            pdb_text = f.read().replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        
        init_lines.append(f"  (function() {{")
        init_lines.append(f"    var el = document.getElementById('viewer_{i}');")
        init_lines.append(f"    var viewer = $3Dmol.createViewer(el, {{backgroundColor: '#0d1117'}});")
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
    html.append("\n".join(init_lines))
    
    html.append("""
</div>
<p class="subheader" style="margin-top: 30px;">
  Pipeline: RFdiffusion2 &rarr; PRISM symmetry &rarr; LigandMPNN sequence &rarr; BindCraft interface<br>
  <em>Powered by PRISM cage design toolkit</em>
</p>
</body></html>""")
    
    output_path.write_text("".join(html))


# ── Main Pipeline ─────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Sequence Design & Interface Optimization Pipeline")
    print("  LigandMPNN × BindCraft × PRISM")
    print("=" * 70)
    
    # Load cage design summary
    summary_path = CAGE_DIR / "design_summary.json"
    if not summary_path.exists():
        print(f"  [ERROR] No cage designs found. Run 04_encapsulin_cage_pipeline.py first.")
        return
    
    with open(summary_path) as f:
        cage_summary = json.load(f)
    
    cage_designs = cage_summary["designs"]
    print(f"\n[1] Found {len(cage_designs)} cage designs from encapsulin pipeline")
    
    all_results = []
    
    # ── Step 1: LigandMPNN sequence design ───────────────────────────
    print(f"\n{'='*70}")
    print(f"  STEP 1: LigandMPNN Sequence Design")
    print(f"{'='*70}")
    
    for idx, cage in enumerate(cage_designs):
        cage_pdb = Path(cage["assembly_pdb"])
        if not cage_pdb.exists():
            print(f"  [SKIP] {cage_pdb.name} not found")
            continue
        
        mpnn_out = OUTPUT_DIR / f"design_{idx}" / "mpnn"
        print(f"\n  [{idx}] Running LigandMPNN on {cage_pdb.name}...")
        print(f"      Homo-oligomer mode, T={MPNN_TEMPERATURE}, {N_MPNN_SEQS} sequences")
        
        mpnn_result = run_mpnn(cage_pdb, mpnn_out, n_seqs=N_MPNN_SEQS,
                              pack_side_chains=True)
        
        n_seqs = len(mpnn_result["sequences"])
        n_packed = len(mpnn_result["packed_pdbs"])
        print(f"      Sequences designed: {n_seqs}")
        print(f"      Packed structures: {n_packed}")
        
        # Show best metrics
        best_conf = 0
        best_seq = ""
        best_metrics = {}
        for j, m in enumerate(mpnn_result["metrics"]):
            conf = m.get("overall_confidence", 0)
            if conf > best_conf:
                best_conf = conf
                best_metrics = m
                if j < len(mpnn_result["sequences"]):
                    best_seq = mpnn_result["sequences"][j]
        
        if best_conf > 0:
            print(f"      Best confidence: {best_conf:.4f}")
            print(f"      Seq recovery: {best_metrics.get('seq_rec', 'N/A')}")
        
        # Choose display PDB (packed if available, else original cage)
        display_pdb = str(cage_pdb)
        if mpnn_result["packed_pdbs"]:
            display_pdb = str(mpnn_result["packed_pdbs"][0])
        elif mpnn_result["backbone_pdbs"]:
            display_pdb = str(mpnn_result["backbone_pdbs"][0])
        
        result = {
            "idx": idx,
            "cage_pdb": str(cage_pdb),
            "display_pdb": display_pdb,
            "n_residues": cage["n_residues"],
            "cage_metrics": cage["metrics"],
            "mpnn_metrics": best_metrics,
            "best_sequence": best_seq,
            "all_sequences": mpnn_result["sequences"],
            "mpnn_packed_pdbs": [str(p) for p in mpnn_result["packed_pdbs"]],
            "mpnn_backbone_pdbs": [str(p) for p in mpnn_result["backbone_pdbs"]],
        }
        
        # Find interface residues
        iface_res = find_interface_residues(cage_pdb)
        result["interface_residues"] = iface_res
        result["n_interface_residues"] = len(iface_res)
        print(f"      Interface residues (chain A): {len(iface_res)} residues")
        if iface_res:
            print(f"      First 10: {iface_res[:10]}")
        
        all_results.append(result)
    
    # ── Step 2: BindCraft interface optimization ─────────────────────
    print(f"\n{'='*70}")
    print(f"  STEP 2: BindCraft Interface Optimization")
    print(f"{'='*70}")
    
    bindcraft_python = BINDCRAFT_ENV / "bin" / "python"
    bindcraft_available = bindcraft_python.exists()
    
    if not bindcraft_available:
        print(f"\n  [INFO] BindCraft environment not ready yet.")
        print(f"         Installation running in background.")
        print(f"         Expected at: {BINDCRAFT_ENV}")
        print(f"         Skipping interface optimization for now.")
        print(f"         Re-run this script after installation completes.")
        
        for result in all_results:
            result["bindcraft_status"] = "pending"
            result["bindcraft_result"] = None
    else:
        for idx, result in enumerate(all_results):
            cage_pdb = Path(result["cage_pdb"])
            bc_out = OUTPUT_DIR / f"design_{idx}" / "bindcraft"
            
            print(f"\n  [{idx}] Running BindCraft on {cage_pdb.name}...")
            
            # Extract subunit pair for interface design
            pair_pdb = bc_out / "subunit_pair_AB.pdb"
            bc_out.mkdir(parents=True, exist_ok=True)
            extract_subunit_pair(cage_pdb, pair_pdb, ("A", "B"))
            
            iface_res = result.get("interface_residues", [])
            if not iface_res:
                print(f"      [SKIP] No interface residues found")
                result["bindcraft_status"] = "skipped"
                result["bindcraft_result"] = None
                continue
            
            bc_result = run_bindcraft(pair_pdb, iface_res, bc_out)
            result["bindcraft_status"] = bc_result.get("status", "unknown")
            result["bindcraft_result"] = bc_result
            
            if bc_result["status"] == "success":
                print(f"      BindCraft designs: {bc_result['n_designs']}")
            else:
                print(f"      BindCraft status: {bc_result['status']}")
                if "reason" in bc_result:
                    print(f"      Reason: {bc_result['reason']}")
    
    # ── Step 3: Summary & ranking ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STEP 3: Design Summary & Ranking")
    print(f"{'='*70}")
    
    print(f"\n  {'Design':>8} {'CavDiam':>8} {'MPNN-Conf':>10} {'SeqRec':>8} "
          f"{'IfaceRes':>10} {'BindCraft':>10}")
    print(f"  {'':>8} {'(nm)':>8} {'':>10} {'':>8} "
          f"{'':>10} {'':>10}")
    print("  " + "-" * 58)
    
    for r in all_results:
        cm = r["cage_metrics"]
        mm = r["mpnn_metrics"]
        conf = mm.get("overall_confidence", 0)
        seq_rec = mm.get("seq_rec", "N/A")
        if isinstance(seq_rec, float):
            seq_rec = f"{seq_rec:.3f}"
        
        print(f"  {r['idx']:>8d} {cm.get('cavity_diameter_nm', 0):>8.1f} "
              f"{conf:>10.4f} {seq_rec:>8} "
              f"{r.get('n_interface_residues', 0):>10d} "
              f"{r.get('bindcraft_status', 'N/A'):>10}")
    
    # ── Step 4: Generate viewer ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STEP 4: Interactive Viewer")
    print(f"{'='*70}")
    
    viewer_path = OUTPUT_DIR / "cage_design_viewer.html"
    generate_final_viewer(all_results, viewer_path)
    print(f"  Viewer: {viewer_path}")
    
    # Write full results JSON
    results_path = OUTPUT_DIR / "pipeline_results.json"
    # Make results JSON-safe
    safe_results = []
    for r in all_results:
        safe = {k: v for k, v in r.items() if k != "bindcraft_result"}
        if r.get("bindcraft_result"):
            safe["bindcraft_result"] = {
                k: v for k, v in r["bindcraft_result"].items()
                if isinstance(v, (str, int, float, bool, type(None), list))
            }
        safe_results.append(safe)
    
    with open(results_path, "w") as f:
        json.dump({
            "pipeline": "LigandMPNN × BindCraft × PRISM",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_designs": len(all_results),
            "mpnn_model": "ligand_mpnn (s25_r010_t300_p.pt)",
            "mpnn_temperature": MPNN_TEMPERATURE,
            "cage_radius_A": 50.0,
            "n_subunits": N_SUBUNITS,
            "designs": safe_results,
        }, f, indent=2)
    print(f"  Results: {results_path}")
    
    # Write FASTA for all best sequences
    fasta_path = OUTPUT_DIR / "all_designed_sequences.fasta"
    with open(fasta_path, "w") as f:
        for r in all_results:
            seq = r.get("best_sequence", "")
            if seq:
                conf = r["mpnn_metrics"].get("overall_confidence", 0)
                f.write(f">cage_design_{r['idx']}_chainA "
                        f"conf={conf:.4f} residues={r['n_residues']}\n")
                # Wrap at 80 chars
                for j in range(0, len(seq), 80):
                    f.write(seq[j:j+80] + "\n")
    print(f"  Sequences: {fasta_path}")
    
    # ── Done ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Files:")
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            rel = p.relative_to(OUTPUT_DIR)
            size_kb = p.stat().st_size / 1024
            print(f"    {str(rel):55s} {size_kb:8.1f} KB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
