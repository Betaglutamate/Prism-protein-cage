#!/usr/bin/env python3
"""
BindCraft Glue Binder Design Pipeline
=======================================
Design small protein binders that bridge adjacent subunits in cage assemblies,
acting as molecular "glue" to stabilise the architecture.

For each cage design:
  1. Identify the closest subunit pair (e.g. chains A–C)
  2. Extract the pair as a 2-chain PDB
  3. Find interface hotspot residues (within 8 Å across chains)
  4. Run BindCraft to hallucinate binders targeting that interface
  5. Score and rank results
"""

import sys, os, json, subprocess, time, csv
import numpy as np
from pathlib import Path
from glob import glob

# ── Configuration ─────────────────────────────────────────────────────
PRISM_ROOT    = Path(__file__).resolve().parent.parent
OUTPUT_DIR    = PRISM_ROOT / "output" / "designed_cages"
BINDCRAFT_DIR = Path("/home/markzurbruegg/library-design/BindCraft")
BINDCRAFT_PY  = Path("/home/markzurbruegg/miniconda3/envs/BindCraft/bin/python")

# Design parameters
BINDER_LENGTHS     = [50, 80]       # min/max residues for designed binder
N_FINAL_DESIGNS    = 50             # total design trajectories attempted
INTERFACE_CUTOFF   = 8.0            # Å — residues within this define the hotspot

# ── Helpers ───────────────────────────────────────────────────────────

def parse_chains(pdb_path: Path) -> dict:
    """Parse PDB into chain_id -> {resnum -> [[x,y,z], ...]}."""
    chain_res = {}
    chain_atoms = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            ch = line[21]
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            resnum = int(line[22:26])
            chain_atoms.setdefault(ch, []).append([x, y, z])
            chain_res.setdefault(ch, {}).setdefault(resnum, []).append([x, y, z])
    return chain_atoms, chain_res


def find_closest_pair(chain_atoms: dict, ref_chain: str = "A") -> tuple:
    """Find the chain closest to ref_chain. Returns (partner_chain, min_dist)."""
    best_chain, best_dist = None, 999.0
    ref = np.array(chain_atoms[ref_chain])
    step_r = max(1, len(ref) // 300)
    for ch, coords in chain_atoms.items():
        if ch == ref_chain:
            continue
        partner = np.array(coords)
        step_p = max(1, len(partner) // 300)
        dists = np.linalg.norm(ref[::step_r, None, :] - partner[None, ::step_p, :], axis=-1)
        mind = float(dists.min())
        if mind < best_dist:
            best_dist = mind
            best_chain = ch
    return best_chain, best_dist


def find_interface_residues(chain_res: dict, chain_atoms: dict,
                            ref_chain: str, partner_chain: str,
                            cutoff: float = INTERFACE_CUTOFF) -> list:
    """Find residues on ref_chain within cutoff of partner_chain."""
    other = np.array(chain_atoms[partner_chain])
    step = max(1, len(other) // 2000)
    iface = []
    for resnum, coords in sorted(chain_res[ref_chain].items()):
        rc = np.array(coords)
        dists = np.linalg.norm(rc[:, None, :] - other[None, ::step, :], axis=-1)
        if dists.min() < cutoff:
            iface.append(resnum)
    return iface


def extract_chain_pair(pdb_path: Path, output_path: Path,
                       chain_a: str, chain_b: str) -> Path:
    """Extract two chains from a PDB, renaming them to A and B."""
    chain_map = {chain_a: "A", chain_b: "B"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pdb_path) as fin, open(output_path, "w") as fout:
        fout.write(f"REMARK   Subunit pair {chain_a}+{chain_b} for BindCraft\n")
        for line in fin:
            if line.startswith(("ATOM", "HETATM", "TER")):
                ch = line[21]
                if ch in chain_map:
                    new_line = line[:21] + chain_map[ch] + line[22:]
                    fout.write(new_line)
        fout.write("END\n")
    return output_path


def write_bindcraft_settings(output_dir: Path, target_pdb: Path,
                             hotspot_residues: list, binder_name: str) -> dict:
    """Write BindCraft JSON settings files. Returns dict of paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Target settings
    hotspot_str = ",".join(str(r) for r in hotspot_residues[:20])
    target_settings = {
        "design_path": str(output_dir / "designs"),
        "binder_name": binder_name,
        "starting_pdb": str(target_pdb),
        "chains": "A",
        "target_hotspot_residues": hotspot_str,
        "lengths": BINDER_LENGTHS,
        "number_of_final_designs": N_FINAL_DESIGNS,
    }
    target_path = output_dir / "target_settings.json"
    target_path.write_text(json.dumps(target_settings, indent=2))

    # Create custom advanced settings with correct AF2 params path
    default_advanced = BINDCRAFT_DIR / "settings_advanced" / "default_4stage_multimer.json"
    with open(default_advanced) as f:
        advanced = json.load(f)
    # Set AF2 params directory to where we downloaded the multimer weights
    advanced["af_params_dir"] = str(BINDCRAFT_DIR)
    advanced["dssp_path"] = str(BINDCRAFT_DIR / "functions" / "dssp")
    advanced["dalphaball_path"] = str(BINDCRAFT_DIR / "functions" / "DAlphaBall.gcc")
    advanced_path = output_dir / "advanced_settings.json"
    advanced_path.write_text(json.dumps(advanced, indent=2))

    # Use BindCraft default filters
    filters_path = BINDCRAFT_DIR / "settings_filters" / "default_filters.json"

    return {
        "target": str(target_path),
        "advanced": str(advanced_path),
        "filters": str(filters_path),
    }


def run_bindcraft(settings: dict, output_dir: Path, design_label: str) -> dict:
    """Run BindCraft with the given settings. Returns result dict."""
    log_path = output_dir / "bindcraft.log"

    cmd = [
        str(BINDCRAFT_PY),
        str(BINDCRAFT_DIR / "bindcraft.py"),
        "--settings", settings["target"],
        "--filters", settings["filters"],
        "--advanced", settings["advanced"],
    ]

    print(f"    CMD: {' '.join(cmd)}")
    print(f"    Log: {log_path}")

    start = time.time()
    try:
        with open(log_path, "w") as log:
            result = subprocess.run(
                cmd,
                cwd=str(BINDCRAFT_DIR),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600 * 4,  # 4 hour timeout per design
            )
        elapsed = time.time() - start

        if result.returncode != 0:
            tail = log_path.read_text()[-1000:]
            print(f"    [WARN] BindCraft exited {result.returncode} after {elapsed/60:.1f}min")
            print(f"    Log tail: {tail[-300:]}")
            return {"status": "failed", "returncode": result.returncode,
                    "elapsed_min": elapsed / 60, "log_tail": tail[-500:]}

        # Parse outputs
        design_dir = output_dir / "designs"
        design_pdbs = sorted(design_dir.glob("**/*.pdb")) if design_dir.exists() else []
        metrics = parse_bindcraft_metrics(design_dir) if design_dir.exists() else []

        print(f"    BindCraft completed in {elapsed/60:.1f}min")
        print(f"    Designs passing filters: {len(design_pdbs)}")

        return {
            "status": "success",
            "n_designs": len(design_pdbs),
            "design_pdbs": [str(p) for p in design_pdbs],
            "metrics": metrics,
            "elapsed_min": elapsed / 60,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"    [TIMEOUT] BindCraft timed out after {elapsed/60:.1f}min")
        return {"status": "timeout", "elapsed_min": elapsed / 60}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def parse_bindcraft_metrics(design_dir: Path) -> list:
    """Parse BindCraft output metrics from CSV files."""
    metrics = []
    for csv_path in design_dir.rglob("*.csv"):
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


# ── Main Pipeline ─────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  BindCraft Glue Binder Design Pipeline")
    print("  Designing binders to stabilise cage subunit interfaces")
    print("=" * 70)

    # Verify BindCraft
    if not BINDCRAFT_PY.exists():
        print(f"[ERROR] BindCraft Python not found: {BINDCRAFT_PY}")
        return
    if not (BINDCRAFT_DIR / "bindcraft.py").exists():
        print(f"[ERROR] BindCraft not found: {BINDCRAFT_DIR}")
        return

    # Load cage designs
    results_path = OUTPUT_DIR / "pipeline_results.json"
    if not results_path.exists():
        print("[ERROR] No pipeline results found. Run 05_mpnn_bindcraft_pipeline.py first.")
        return

    with open(results_path) as f:
        pipeline = json.load(f)

    designs = pipeline["designs"]
    print(f"\n[1] Loaded {len(designs)} cage designs")

    # ── Step 1: Analyse interfaces & prepare targets ─────────────────
    print(f"\n{'='*70}")
    print(f"  STEP 1: Interface Analysis & Target Preparation")
    print(f"{'='*70}")

    bindcraft_jobs = []

    for design in designs:
        idx = design["idx"]
        cage_pdb = Path(design["cage_pdb"])
        if not cage_pdb.exists():
            print(f"  [SKIP] Design {idx}: PDB not found")
            continue

        print(f"\n  Design {idx}: {cage_pdb.name}")
        chain_atoms, chain_res = parse_chains(cage_pdb)

        # Find closest subunit pair
        partner_chain, min_dist = find_closest_pair(chain_atoms, "A")
        print(f"    Closest pair: A–{partner_chain} ({min_dist:.1f} Å)")

        # Find interface residues
        iface_res = find_interface_residues(chain_res, chain_atoms,
                                            "A", partner_chain, INTERFACE_CUTOFF)
        print(f"    Interface residues (chain A): {len(iface_res)}")
        if iface_res:
            print(f"    Hotspots: {iface_res[:15]}{'...' if len(iface_res) > 15 else ''}")

        if len(iface_res) < 3:
            print(f"    [SKIP] Too few interface residues — subunits too far apart")
            continue

        # Extract subunit pair
        bc_dir = OUTPUT_DIR / f"design_{idx}" / "bindcraft"
        pair_pdb = bc_dir / f"subunit_pair_A{partner_chain}.pdb"
        extract_chain_pair(cage_pdb, pair_pdb, "A", partner_chain)
        print(f"    Extracted pair: {pair_pdb.name}")

        # Write BindCraft settings
        binder_name = f"cage{idx}_glue"
        settings = write_bindcraft_settings(bc_dir, pair_pdb, iface_res, binder_name)
        print(f"    Settings written")

        bindcraft_jobs.append({
            "idx": idx,
            "cage_pdb": str(cage_pdb),
            "pair_pdb": str(pair_pdb),
            "partner_chain": partner_chain,
            "interface_residues": iface_res,
            "min_dist_A": min_dist,
            "settings": settings,
            "bc_dir": bc_dir,
        })

    print(f"\n  Total BindCraft jobs: {len(bindcraft_jobs)}")

    # ── Step 2: Run BindCraft ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STEP 2: Running BindCraft ({len(bindcraft_jobs)} jobs)")
    print(f"{'='*70}")

    all_results = []
    for job in bindcraft_jobs:
        idx = job["idx"]
        bc_dir = job["bc_dir"]

        print(f"\n  [{idx}] Running BindCraft for cage design {idx}")
        print(f"      Target: A–{job['partner_chain']} interface ({job['min_dist_A']:.1f} Å)")
        print(f"      Hotspots: {len(job['interface_residues'])} residues")
        print(f"      Binder length: {BINDER_LENGTHS[0]}–{BINDER_LENGTHS[1]} residues")
        print(f"      Designs: {N_FINAL_DESIGNS}")

        bc_result = run_bindcraft(job["settings"], bc_dir, f"design_{idx}")

        result = {
            **job,
            "bindcraft_result": bc_result,
        }
        # Remove non-serializable Path
        result["bc_dir"] = str(result["bc_dir"])
        all_results.append(result)

        status = bc_result.get("status", "unknown")
        if status == "success":
            print(f"      Result: {bc_result['n_designs']} binders designed")
        else:
            print(f"      Result: {status}")

    # ── Step 3: Score & Rank ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STEP 3: Scoring & Ranking")
    print(f"{'='*70}")

    for result in all_results:
        idx = result["idx"]
        bc = result["bindcraft_result"]
        if bc.get("status") != "success" or not bc.get("metrics"):
            print(f"  Design {idx}: {bc.get('status', 'N/A')} — no metrics")
            continue

        metrics = bc["metrics"]
        print(f"\n  Design {idx}: {len(metrics)} binder designs scored")

        # Sort by composite score: iPTM * pLDDT / pAE_interaction
        scored = []
        for i, m in enumerate(metrics):
            iptm = m.get("Average_i_pTM", m.get("iptm", 0))
            plddt = m.get("Average_pLDDT", m.get("plddt", 0))
            pae = m.get("Average_i_pAE", m.get("pae_interaction", 99))
            if pae > 0:
                score = iptm * plddt / pae
            else:
                score = 0
            scored.append((score, i, m))

        scored.sort(reverse=True)

        print(f"  {'Rank':>4} {'Score':>8} {'iPTM':>6} {'pLDDT':>6} {'pAE':>6}")
        print(f"  {'':>4} {'':>8} {'':>6} {'':>6} {'':>6}")
        for rank, (score, i, m) in enumerate(scored[:5]):
            iptm = m.get("Average_i_pTM", m.get("iptm", 0))
            plddt = m.get("Average_pLDDT", m.get("plddt", 0))
            pae = m.get("Average_i_pAE", m.get("pae_interaction", 0))
            print(f"  {rank+1:>4} {score:>8.2f} {iptm:>6.3f} {plddt:>6.1f} {pae:>6.1f}")

    # ── Save results ─────────────────────────────────────────────────
    results_out = OUTPUT_DIR / "bindcraft_glue_results.json"
    safe_results = []
    for r in all_results:
        safe = {}
        for k, v in r.items():
            if k == "settings":
                safe[k] = v
            elif k == "bindcraft_result":
                bc = v
                safe[k] = {kk: vv for kk, vv in bc.items()
                           if isinstance(vv, (str, int, float, bool, type(None), list))}
            elif isinstance(v, (str, int, float, bool, type(None), list)):
                safe[k] = v
        safe_results.append(safe)

    with open(results_out, "w") as f:
        json.dump({
            "pipeline": "BindCraft Glue Binder Design",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_jobs": len(all_results),
            "binder_lengths": BINDER_LENGTHS,
            "n_designs_per_job": N_FINAL_DESIGNS,
            "interface_cutoff_A": INTERFACE_CUTOFF,
            "results": safe_results,
        }, f, indent=2)
    print(f"\n  Results saved: {results_out}")

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
