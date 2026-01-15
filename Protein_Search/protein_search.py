import argparse
import os
import sys
import subprocess
import re
import time
import json
import numpy as np

# --- Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, "data", "protein")
BLAST_DB = os.path.join(DATA_DIR, "blast_db", "protein_swissprot")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "protein")
BIN_DIR = os.path.join(ROOT_DIR, "bin")

# Defaults - will be overridden by config file
PARAMS = {
    "lsh":       {"k": 4, "L": 5, "w": 4.0},
    "hypercube": {"kproj": 14, "w": 4.0, "M": 10, "probes": 2},
    "ivfflat":   {"kclusters": 50, "nprobe": 5},
    "ivfpq":     {"kclusters": 50, "nprobe": 5, "M": 16, "nbits": 8},
    "neural":    {"m": 100, "T": 10, "k": 10, "epochs": 10, "layers": 3, "nodes": 64, "lm": 1.0} 
}
# Global options from config
GLOBAL_OPTS = {
    "R": 0.0,
    "range": False,
    "seed": 1,
    "metric": "l2"
}

def get_unique_filename(filepath):
    """
    If filepath exists, appends _1, _2, etc. until a unique name is found.
    """
    if not os.path.exists(filepath):
        return filepath
    
    base, ext = os.path.splitext(filepath)
    counter = 1
    while True:
        new_path = f"{base}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def load_ids(path):
    if not os.path.exists(path): return []
    with open(path, 'r') as f:
        return [l.strip() for l in f]

def run_blast_ground_truth(db_fasta, query_fasta, top_n):
    """Runs BLASTp and returns a dict: QueryID -> {ids: Set, table: Dict[id, identity]}"""
    os.makedirs(os.path.dirname(BLAST_DB), exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, "blast_results.tsv")
    
    # 1. Make DB
    if not os.path.exists(f"{BLAST_DB}.pin"):
        print("[BLAST] Building Database...")
        subprocess.run(["makeblastdb", "-in", db_fasta, "-dbtype", "prot", "-out", BLAST_DB], 
                       check=True, stdout=subprocess.DEVNULL)

    # 2. Run Search
    if not os.path.exists(out_file):
        print("[BLAST] Running Alignment (this may take time)...")
        # outfmt 6: qseqid sseqid pident ...
        cmd = ["blastp", "-query", query_fasta, "-db", BLAST_DB, 
               "-out", out_file, "-outfmt", "6", "-evalue", "1.0", 
               "-max_target_seqs", str(top_n * 5)] # Get more to filter down to N unique
        subprocess.run(cmd, check=True)

    # 3. Parse
    print("[BLAST] Parsing Results...")
    gt = {}
    with open(out_file, 'r') as f:
        for line in f:
            parts = line.split('\t')
            if len(parts) < 3: continue
            qid, sid, identity = parts[0], parts[1], float(parts[2])
            
            if qid not in gt: 
                gt[qid] = []
            
            # Store tuple (sid, identity)
            gt[qid].append((sid, identity))

    # Keep Top-N unique per query
    final_gt = {}
    for qid, hits in gt.items():
        # Hits are usually sorted by bitscore by blast, so just take unique top N
        seen = set()
        top_unique = []
        for sid, ident in hits:
            if sid not in seen:
                seen.add(sid)
                top_unique.append((sid, ident))
            if len(top_unique) == top_n: break
            
        final_gt[qid] = {
            "ids_set": set(h[0] for h in top_unique),
            "ident_map": {h[0]: h[1] for h in top_unique}
        }
    return final_gt

def run_method(method, args, db_fvecs, q_fvecs, metric="l2"):
    res_path = os.path.join(OUTPUT_DIR, method, f"{method}_res.txt")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    
    print(f"[{method.upper()}] Running with {metric.upper()}...")
    
    # Global args preparation
    range_str = "true" if GLOBAL_OPTS["range"] else "false"

    if method == "neural":
        # Call Python Neural LSH
        p = PARAMS["neural"]
        # Append metric to filename to ensure model matches metric (e.g. protein_index_l2_model.pth)
        idx_prefix = os.path.join(ROOT_DIR, "cache", "models", f"protein_index_{metric}")
        
        # Build if missing (calling nlsh_build.py)
        if not os.path.exists(f"{idx_prefix}_model.pth"):
             print(f"[{method.upper()}] Building Neural Index with {metric}...")
             cmd = ["python3", os.path.join(ROOT_DIR, "NLSH", "nlsh_build.py"),
                    "-d", db_fvecs, "-i", idx_prefix, "-type", "generic",
                    "--m", str(p['m']), "--knn", str(p['k']), "--epochs", str(p['epochs']),
                    "--nodes", str(p['nodes']), "--layers", str(p['layers']),
                    "--metric", metric]
             subprocess.run(cmd, check=True)

        # Search (calling nlsh_search.py)
        cmd = ["python3", os.path.join(ROOT_DIR, "NLSH", "nlsh_search.py"),
               "-d", db_fvecs, "-q", q_fvecs, "-i", idx_prefix, "-o", res_path,
               "-type", "generic", "-N", str(args.N), "-T", str(p['T']),
               "-metric", metric]
        # Note: Neural search internally uses cosine if model trained on it, but here we just pass data
        # If your neural LSH script needs metric flag, add it here. 
        # Assuming current nlsh_search.py uses L2/Cosine based on training or default.
        subprocess.run(cmd, check=True)
        
    else:
        # Call C++ Binary ("search")
        binary = os.path.join(BIN_DIR, "search")
        p = PARAMS[method]
        
        cmd = [binary, "-algo", method, "-d", db_fvecs, "-q", q_fvecs, "-o", res_path, 
               "-type", "protein", "-N", str(args.N), "-threads", "1", "-metric", metric,
               "-R", str(GLOBAL_OPTS["R"]), "-range", range_str, "-seed", str(GLOBAL_OPTS["seed"])] 

        # Map params to CLI args
        if method == "lsh":
            cmd.extend(["-k", str(p['k']), "-L", str(p['L']), "-w", str(p['w'])])
        elif method == "hypercube":
            cmd.extend(["-kproj", str(p['kproj']), "-M", str(p['M']), "-probes", str(p['probes']), "-w", str(p['w'])])
        elif method == "ivfflat":
            cmd.extend(["-kclusters", str(p['kclusters']), "-nprobe", str(p['nprobe'])])
        elif method == "ivfpq":
            # FIXED: args_parser.cpp expects -M and -nbits, not -pq_M / -pq_nbits
            cmd.extend(["-kclusters", str(p['kclusters']), "-nprobe", str(p['nprobe']), 
                        "-M", str(p['M']), "-nbits", str(p['nbits'])])
        
        # Capture output may be printed to stdout, but result_writer saves to res_path
        subprocess.run(cmd, check=True)
        
    return res_path

def parse_ann(filepath, db_ids, q_ids):
    """Parses output. Maps Index -> Protein String ID."""
    results = {}
    metrics = {"qps": 0.0, "time": 0.0}
    
    if not os.path.exists(filepath): return metrics, results
    
    current_qid = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Metrics (From Summary part of file)
            if "QPS:" in line:
                metrics["qps"] = float(line.split(":")[-1].strip())
            # Sum approx time from metric calc later
            if "tApproxAvg=" in line: # Or calculate manually from execution time
                pass 
            
            # Results
            if line.startswith("Query:"):
                # Query: <ID> or Query: <Int Index>
                q_val = line.split(":")[-1].strip()
                if q_val.isdigit():
                    idx = int(q_val)
                    if idx < len(q_ids): current_qid = q_ids[idx]
                else:
                    current_qid = q_val
                if current_qid: results[current_qid] = []
                
            elif line.startswith("Nearest neighbor") and current_qid:
                # Nearest neighbor-1: <Int Index>
                idx = int(line.split(":")[-1].strip())
                if idx < len(db_ids):
                    results[current_qid].append({"id": db_ids[idx], "dist": 0.0})
            
            elif line.startswith("distanceApproximate:") and current_qid and results[current_qid]:
                dist = float(line.split(":")[-1].strip())
                results[current_qid][-1]["dist"] = dist
                
    # Calculate Total Time and QPS manually if not parsed
    # We will rely on execution loop timing for summary if file missing
    return metrics, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, help="DB .npy file prefix (without extension if possible)")
    parser.add_argument("-q", required=True, help="Query .npy file prefix")
    parser.add_argument("-db_fasta", required=True)
    parser.add_argument("-q_fasta", required=True)
    parser.add_argument("-o", required=True, help="Report output file")
    parser.add_argument("-method", default="all")
    parser.add_argument("-N", type=int, default=50) 
    parser.add_argument("-config", help="Path to JSON config file to override PARAMS", default=None)
    args = parser.parse_args()

    # --- 0. Load Configuration ---
    active_metric = "l2"
    
    if args.config:
        print(f"[Config] Loading parameters from {args.config}...")
        try:
            with open(args.config, 'r') as f:
                config_json = json.load(f)
                
                # Global overrides
                if "metric" in config_json:
                    active_metric = config_json["metric"]
                    GLOBAL_OPTS["metric"] = active_metric
                    print(f"  > Metric set to: {active_metric}")
                
                # Check for R, range, seed in top level
                for key in ["R", "range", "seed"]:
                    if key in config_json:
                        GLOBAL_OPTS[key] = config_json[key]
                        print(f"  > Updated {key}: {GLOBAL_OPTS[key]}")

                for algo_key, params_dict in config_json.items():
                    if algo_key in PARAMS:
                        PARAMS[algo_key].update(params_dict)
                        print(f"  > Updated {algo_key}: {PARAMS[algo_key]}")
        except Exception as e:
            print(f"[Error] Failed to load config: {e}")
            sys.exit(1)

    # 1. File Sync (NPY <-> FVECS <-> IDS)
    # Allow input as "data/protein/protein_db.npy" or just "data/protein/protein_db"
    db_base = os.path.splitext(args.d)[0]
    q_base = os.path.splitext(args.q)[0]
    
    # Python needs .npy (via implicit internal loading usually) but maps via _ids.txt
    db_ids = load_ids(f"{db_base}_ids.txt")
    q_ids = load_ids(f"{q_base}_ids.txt")
    
    # C++ needs .fvecs
    db_fvecs = f"{db_base}.fvecs"
    q_fvecs = f"{q_base}.fvecs"

    if not db_ids or not q_ids:
        print(f"[Error] ID mapping files missing ({db_base}_ids.txt). Run protein_embed.py first.")
        sys.exit(1)

    # 2. Ground Truth
    blast_gt = run_blast_ground_truth(args.db_fasta, args.q_fasta, args.N)

    # 3. Execution
    methods = ["lsh", "hypercube", "ivfflat", "ivfpq", "neural"] if args.method == "all" else args.method.split(",")
    ann_results = {}
    
    for m in methods:
        start_t = time.time()
        path = run_method(m, args, db_fvecs, q_fvecs, active_metric)
        total_t_sec = time.time() - start_t
        
        metrics, data = parse_ann(path, db_ids, q_ids)
        
        # Calculate Recall
        recall_sum = 0
        valid = 0
        for qid, neighbors in data.items():
            if qid in blast_gt:
                ann_ids = set(n['id'] for n in neighbors[:args.N])
                truth_ids = blast_gt[qid]["ids_set"]
                
                if not truth_ids: continue
                
                hits = len(ann_ids.intersection(truth_ids))
                recall_sum += hits / len(truth_ids)
                valid += 1
        
        avg_recall = recall_sum / valid if valid > 0 else 0
        
        # Approximate QPS if not in file
        if metrics["qps"] == 0.0 and len(q_ids) > 0:
            metrics["qps"] = len(q_ids) / total_t_sec if total_t_sec > 0 else 0
            
        # Approx time per query
        metrics["time_pq"] = total_t_sec / len(q_ids) if len(q_ids) > 0 else 0
        
        ann_results[m] = {"metrics": metrics, "data": data, "recall": avg_recall}

    # 4. Report Generation
    # Adjust filename if needed to respect metric
    if "final_report.txt" in os.path.basename(args.o) and active_metric != "l2":
        base_dir = os.path.dirname(args.o)
        args.o = os.path.join(base_dir, f"final_report_{active_metric}.txt")

    # Ensure unique filename to prevent overwrites
    args.o = get_unique_filename(args.o)
    print(f"[Info] Final Output Report: {args.o}")

    print(f"\n[Report] Generating {args.o}...")
    with open(args.o, 'w') as f:
        f.write("Protein Homology Search Report\n")
        f.write("==============================\n\n")

        # --- Configuration Header ---
        f.write("[0] Configuration Parameters\n")
        f.write("-" * 85 + "\n")
        f.write(f"Metric: {active_metric}\n")
        f.write(f"Global Seed: {GLOBAL_OPTS.get('seed', 1)}\n")
        f.write(f"Range Search: {GLOBAL_OPTS.get('range', False)} (R={GLOBAL_OPTS.get('R', 0.0)})\n")
        f.write("\nAlgorithm Parameters:\n")
        for m in methods:
            if m in PARAMS:
                params_str = ", ".join([f"{k}={v}" for k,v in PARAMS[m].items()])
                f.write(f"  - {m.upper():<10}: {params_str}\n")
        f.write("-" * 85 + "\n\n")
        
        f.write(f"[1] Summary Comparison (Ground Truth: BLAST, N={args.N})\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'Method':<15} | {'Time/query (s)':<15} | {'QPS':<10} | {'Recall@N':<10}\n")
        f.write("-" * 85 + "\n")
        
        for m in methods:
            stats = ann_results[m]
            f.write(f"{m.upper():<15} | {stats['metrics']['time_pq']:<15.4f} | {stats['metrics']['qps']:<10.1f} | {stats['recall']:<10.4f}\n")
        
        f.write(f"{'BLAST (Ref)':<15} | {'-':<15} | {'-':<10} | {'1.0000'}\n")
        f.write("-" * 85 + "\n\n")

        f.write(f"[2] Detailed Top-N Analysis (All Queries)\n")
        
        # Analyze all queries that have ground truth (or all queries if GT not strict)
        valid_queries = [qid for qid in q_ids if qid in blast_gt]
        
        for qid in valid_queries:
            blast_info = blast_gt[qid]
            f.write(f"\nQUERY PROTEIN: {qid}\n")
            
            for m in methods:
                f.write(f"\nMethod: {m.upper()}\n")
                f.write("-" * 115 + "\n")
                f.write(f"{'Rank':<5} | {'Neighbor ID':<25} | {'L2 Dist':<10} | {'BLAST %':<8} | {'In BLAST Top-N?':<18} | {'Bio Comment'}\n")
                f.write("-" * 115 + "\n")
                
                neighbors = ann_results[m]["data"].get(qid, []) 
                # Ensure we respect N
                display_neighbors = neighbors[:args.N] if neighbors else []
                
                for i, n in enumerate(display_neighbors):
                    nid = n['id']
                    dist = n['dist']
                    in_blast = nid in blast_info["ids_set"]
                    pident = blast_info["ident_map"].get(nid, 0.0)
                    
                    comment = ""
                    in_blast_str = "Yes" if in_blast else "No"
                    
                    if in_blast:
                        if pident < 30.0:
                            comment = "Remote Homolog (Twilight Zone)"
                        else:
                            comment = "Known Homolog"
                    else:
                        if dist < 0.2: 
                            comment = "Novel Candidate?"
                        else:
                            comment = "Likely False Positive"
                    
                    f.write(f"{i+1:<5} | {nid:<25} | {dist:<10.4f} | {pident:<8.1f} | {in_blast_str:<18} | {comment}\n")
            f.write("=" * 115 + "\n")

if __name__ == "__main__":
    main()