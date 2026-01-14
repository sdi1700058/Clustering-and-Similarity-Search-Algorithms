import argparse
import os
import sys
import subprocess
import numpy as np
import time
import re
import json

# Add parent dir to path to find NLSH and bin
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

# --- CONFIGURATION ---
DATA_DIR = os.path.join(ROOT_DIR, "data", "protein")
BLAST_DB_DIR = os.path.join(DATA_DIR, "blast_db")
OUTPUT_BASE = os.path.join(ROOT_DIR, "output", "protein")
FIG_DEST = os.path.join(OUTPUT_BASE, "figures")

# Ensure Dirs match
for d in [DATA_DIR, BLAST_DB_DIR, OUTPUT_BASE, FIG_DEST]:
    os.makedirs(d, exist_ok=True)

CONFIG_FILE = os.path.join(ROOT_DIR, "protein_config.json")

def load_config():
    defaults = {
        "lsh":       {"k": 6, "L": 8, "w": 4.0},
        "hypercube": {"kproj": 12, "w": 4.0, "M": 5000, "probes": 2},
        "ivfflat":   {"kclusters": 100, "nprobe": 5},
        "ivfpq":     {"kclusters": 100, "nprobe": 5, "M": 16, "nbits": 8},
        "neural":    {"m": 400, "T": 50, "k": 10, "epochs": 15, "layers": 3, "nodes": 128} 
    }
    
    # Merge nested common params from algorithm section to global defaults
    # This allows config files to specify "metric" or "range" at top level OR inside algo sections
    # However the script expects them as keys in 'defaults' if we just iterate items
    # We refactor loop to support deep update if needed, but for now we follow the structure
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                user_config = json.load(f)
                
                # Check for top-level keys like "metric", "range"
                # If they exist, we must add them to 'defaults' so run_cpp_algo can see them
                for key in ["metric", "range", "R", "seed"]:
                    if key in user_config:
                         # We store them in a dummy section or just attach to algo sections?
                         # run_cpp_algo reads 'params' passed to it.
                         # We iterate ALL methods. We should inject these common params into EACH method config.
                         val = user_config[key]
                         for algo in defaults:
                             defaults[algo][key] = val

                for k, v in user_config.items():
                    if k in defaults and isinstance(v, dict):
                         defaults[k].update(v)
                        
            print(f"[Config] Loaded configuration from {CONFIG_FILE}")
        except Exception as e:
            print(f"[Config] Error loading json: {e}. Using defaults.")
    return defaults

DEFAULTS = load_config()

FINAL_STATS = []

def get_unique_filename(base_path, name, ext=".txt"):
    counter = 1
    # Ensure directory exists
    os.makedirs(base_path, exist_ok=True)
    while True:
        candidate = os.path.join(base_path, f"{name}_{counter}{ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1

def write_fvecs(filename, data):
    """Writes vectors to .fvecs format for C++ binary."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    print(f"[IO] Writing {filename} ({len(data)} vectors)...")
    n, d = data.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(data[i].tobytes())

def ensure_fvecs(npy_path):
    """
    Ensures a corresponding .fvecs file exists in the SAME directory.
    Returns the path to the .fvecs file.
    """
    base_name = os.path.basename(npy_path)
    # Remove .npy if present
    if base_name.endswith(".npy"): 
        base_name = base_name[:-4]
    # Remove .dat if present
    if base_name.endswith(".dat"): 
        base_name = base_name[:-4]
    
    fvecs_name = f"{base_name}.fvecs"
    fvecs_path = os.path.join(DATA_DIR, fvecs_name)

    if os.path.exists(fvecs_path):
        # Optional: Check timestamp if you want strictly fresh data, 
        # but for large files invalidating cache is expensive. 
        # Here we trust the existence.
        print(f"[Cache] Found existing fvecs: {fvecs_path}")
        return fvecs_path

    print(f"[IO] Converting {npy_path} -> {fvecs_path}")
    try:
        data = np.load(npy_path, allow_pickle=True)
    except:
        print(f"[Error] Could not load {npy_path}")
        sys.exit(1)

    write_fvecs(fvecs_path, data)
    return fvecs_path

# --- BLAST ---
def run_blast(db_fasta, query_fasta, top_n):
    print("[BLAST] Preparing Ground Truth...")
    
    db_name = os.path.join(BLAST_DB_DIR, "protein_swissprot")
    out_table = os.path.join(OUTPUT_BASE, "blast_results.tsv")

    # 1. Make Blast DB if missing (checking .pin file)
    if not os.path.exists(f"{db_name}.pin"):
        cmd = ["makeblastdb", "-in", db_fasta, "-dbtype", "prot", "-out", db_name]
        subprocess.run(cmd, check=True)
    
    # 2. Run BlastP
    if not os.path.exists(out_table):
        print("[BLAST] Running BlastP...")
        cmd = [
            "blastp", "-query", query_fasta, "-db", db_name,
            "-out", out_table, "-outfmt", "6 qseqid sseqid score",
            "-max_target_seqs", str(top_n)
        ]
        subprocess.run(cmd, check=True)
    else:
        print("[BLAST] Using existing blast_results.tsv")

    # 3. Parse and return if needed (omitted for brevity if not strictly used by C++)
    return {}

# --- C++ RUNNER ---
def run_cpp_algo(method, db_fvecs, q_fvecs, args):
    out_dir = os.path.join(OUTPUT_BASE, method)
    os.makedirs(out_dir, exist_ok=True)
    
    res_file = get_unique_filename(out_dir, f"{method}_res")
    binary = os.path.join(ROOT_DIR, "bin", "search")

    cmd = [
        binary, "-algo", method,
        "-d", db_fvecs, "-q", q_fvecs,
        "-o", res_file, "-type", "protein",
        "-N", str(args.N),
        "-metric", "cosine" # Enforced for protein embeddings
    ]
    
    # Add generic params
    if method in DEFAULTS:
        params = DEFAULTS[method]
        for k, v in params.items():
            cmd.extend([f"-{k}", str(v)])

    print(f"\n--- Processing {method} ---")
    print(f"[C++] Running {method} -> {os.path.basename(res_file)}")
    try:
        subprocess.run(cmd, check=True)
        parse_ann_results(res_file, method_name=method)
    except subprocess.CalledProcessError as e:
        print(f"[Error] C++ {method} failed: {e}")

# --- NEURAL RUNNER ---
def run_neural(db_fvecs, q_fvecs, args):
    out_dir = os.path.join(OUTPUT_BASE, "neural")
    os.makedirs(out_dir, exist_ok=True)
    res_file = get_unique_filename(out_dir, "neural_res")

    # Smart Caching: Model Signature
    neural_cfg = DEFAULTS["neural"]
    signature = (f"neural_m{neural_cfg['m']}_k{neural_cfg['k']}"
                 f"_ep{neural_cfg['epochs']}_L{neural_cfg['layers']}_n{neural_cfg['nodes']}")
    
    # Prefix for model files
    cache_models = os.path.join(ROOT_DIR, "cache", "models")
    os.makedirs(cache_models, exist_ok=True)
    index_prefix = os.path.join(cache_models, signature)
    
    # The expected model file is usually {prefix}_model.pth
    model_file = f"{index_prefix}_model.pth"
    
    build_needed = True
    if os.path.exists(model_file):
        print(f"[Neural] Found cached model: {model_file}")
        print("[Neural] Skipping build (using cache).")
        build_needed = False
    
    # 1. Train/Build (only if needed)
    if build_needed:
        print(f"[Neural] Building index with signature {signature}...")
        build_script = os.path.join(ROOT_DIR, "NLSH", "nlsh_build.py")
        cmd_build = [
            "python3", build_script,
            "-d", db_fvecs,
            "-i", index_prefix,
            "-type", "protein",
            "--knn", str(neural_cfg["k"]),
            "--m", str(neural_cfg["m"]),
            "--epochs", str(neural_cfg["epochs"]),
            "--layers", str(neural_cfg["layers"]),
            "--nodes", str(neural_cfg["nodes"])
        ]
        subprocess.run(cmd_build, check=True)

    # 2. Search
    print("[Neural] Searching...")
    search_script = os.path.join(ROOT_DIR, "NLSH", "nlsh_search.py")
    cmd_search = [
        "python3", search_script,
        "-d", db_fvecs,
        "-q", q_fvecs,
        "-i", index_prefix,
        "-o", res_file,
        "-type", "protein",
        "-N", str(args.N),
        "-T", str(neural_cfg["T"]),
        "-metric", "cosine"
    ]
    subprocess.run(cmd_search, check=True)
    
    # Move plots if any were generated in NLSH/fig
    src_fig_dir = os.path.join(ROOT_DIR, "NLSH", "fig")
    if os.path.exists(src_fig_dir):
        move_count = 0
        for f in os.listdir(src_fig_dir):
            if f.endswith(".png"):
                 # Only move if filename looks generated by us (skip subdirs)
                 if os.path.isfile(os.path.join(src_fig_dir, f)):
                     os.rename(os.path.join(src_fig_dir, f), os.path.join(FIG_DEST, f))
                     move_count += 1
        if move_count > 0:
            print(f"[Neural] Moved {move_count} plots to {FIG_DEST}")
            
    parse_ann_results(res_file, method_name="Neural")

def notify_completion(msg="Task Completed"):
    print('\a') # Beep
    try:
        if sys.platform.startswith('linux'):
            # Using notify-send if available
            subprocess.run(['notify-send', 'Protein Search', msg], stderr=subprocess.DEVNULL)
    except: pass

def parse_ann_results(filename, method_name):
    """
    Parses the output text file to extract Average AF, Recall, QPS.
    Adds to FINAL_STATS for comparison table.
    """
    stats = {}
    if not os.path.exists(filename):
        return

    with open(filename, 'r') as f:
        content = f.read()
        
        # Regex parsing
        # Support both ':' (C++ default) and '=' (Legacy/Alternative)
        af = re.search(r"Average AF[:=]\s*([0-9.]+)", content)
        if not af: af = re.search(r"AF[:=]\s*([0-9.]+)", content)
        
        rec = re.search(r"Recall@\d+[:=]\s*([0-9.]+)", content)
        
        qps = re.search(r"QPS[:=]\s*([0-9.]+)", content)
        
        if af: stats['AF'] = float(af.group(1))
        if rec: stats['Recall'] = float(rec.group(1))
        if qps: stats['QPS'] = float(qps.group(1))
        
    FINAL_STATS.append({
        'Method': method_name,
        'Recall': stats.get('Recall', 0.0),
        'QPS': stats.get('QPS', 0.0),
        'AF': stats.get('AF', 0.0),
        'File': f"file://{os.path.abspath(filename)}"
    })
    
    print(f"[Stats] {method_name} -> Recall: {stats.get('Recall', 0):.4f} | QPS: {stats.get('QPS', 0):.2f}")


def wizard():
    print("\n=== Protein Search Interactive Mode ===")
    print("No arguments provided. Entering interactive mode.\n")
    
    args = argparse.Namespace()
    
    # Ask for Datasets (with defaults)
    def ask(prompt, default):
        val = input(f"{prompt} [default: {default}]: ")
        return val.strip() if val.strip() else default

    args.d = "data/protein/protein_db_norm.npy" # Fixed for now as per Makefile
    args.q = "data/protein/targets_vectors_norm.npy"
    
    # Algorithms
    print("\nSelect Algorithms to run (comma separated, or 'all')")
    print(f"Available: {', '.join(DEFAULTS.keys())}")
    algo_in = ask("Algorithms", "all")
    args.method = algo_in
    
    # Params
    while True:
        try:
            val = ask("Number of Neighbors (N)", "50")
            args.N = int(val)
            break
        except ValueError:
            print("[Error] Please enter a valid integer.")
    
    # Placeholders for integration 
    args.db_fasta = "data/protein/swissprot_50k.fasta"
    args.q_fasta = "data/protein/targets.fasta"
    args.pfam = "data/protein/targets.pfam_map.tsv"
    args.o = "output/protein/final_report.txt"
    
    return args

def print_comparison_table():
    if not FINAL_STATS: return
    
    print("\n" + "="*80)
    print(f"{'ALGORITHM PERFORMANCE SUMMARY':^80}")
    print("="*80)
    print(f"{'METHOD':<15} | {'RECALL':<10} | {'QPS':<10} | {'AF':<10} | {'OUTPUT FILE'}")
    print("-" * 80)
    
    # Sort by Recall descending
    sorted_stats = sorted(FINAL_STATS, key=lambda x: x['Recall'], reverse=True)
    
    for s in sorted_stats:
        print(f"{s['Method']:<15} | {s['Recall']:<10.4f} | {s['QPS']:<10.2f} | {s['AF']:<10.4f} | {s['File']}")
    print("="*80 + "\n")

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Protein Similarity Search Benchmark")
    parser.add_argument("-d", help="Database embeddings (.npy)")
    parser.add_argument("-q", help="Query embeddings (.npy)")
    parser.add_argument("-db_fasta", help="Database FASTA")
    parser.add_argument("-q_fasta", help="Query FASTA")
    parser.add_argument("-pfam", help="Pfam mapping")
    parser.add_argument("-o", help="Output report")
    parser.add_argument("-method", default="all", help="all, or comma-separated: lsh,hypercube,ivfflat,ivfpq,neural")
    parser.add_argument("-N", type=int, default=50, help="Number of neighbors")
    
    # Only parse args if command line arguments are present, otherwise run wizard
    if len(sys.argv) == 1:
        args = wizard()
    else:
        args = parser.parse_args()
        
        # If args are absent in CLI mode but not Wizard, apply defaults for simplicity if allowed
        if not args.d: args.d = "data/protein/protein_db_norm.npy"
        if not args.q: args.q = "data/protein/targets_vectors_norm.npy"

    # Integrity Check
    if not os.path.exists(args.d) or not os.path.exists(args.q):
        print(f"[Error] Input files not found:\n  DB: {args.d}\n  Query: {args.q}\nRun 'make protein_embed' first.")
        sys.exit(1)

    try:
        # 1. Convert/Ensure FVECS for C++
        db_fvecs = ensure_fvecs(args.d)
        q_fvecs  = ensure_fvecs(args.q)

        # 2. Ground Truth (BLAST)
        # We can run it just to ensure metric/GT exists, 
        # but pure C++ execution manages its own numeric ground truth via BruteForce.
        # run_blast(args.db_fasta, args.q_fasta, args.N)

        print("\n[Benchmark] Starting execution...\n")

        methods = []
        if args.method == "all":
            methods = list(DEFAULTS.keys())
        else:
            methods = args.method.split(",")

        # 3. Execute Algorithms
        for m in methods:
            m = m.strip()
            if m == "neural":
                run_neural(db_fvecs, q_fvecs, args)
            elif m in ["lsh", "hypercube", "ivfflat", "ivfpq"]:
                run_cpp_algo(m, db_fvecs, q_fvecs, args)
            else:
                print(f"[Warning] Unknown method: {m}")

        # 4. Finalize
        print_comparison_table()
        notify_completion("Benchmark Finished Successfully")

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user. Exiting...")
        print_comparison_table() # Show what we got so far
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        # import traceback; traceback.print_exc() # Uncomment for debug
        sys.exit(1)

if __name__ == "__main__":
    main()
