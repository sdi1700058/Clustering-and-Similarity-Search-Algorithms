import argparse
import numpy as np
import torch
import time
import pickle
import sys
import os
import matplotlib.pyplot as plt

# Local imports
from data_parser import get_dataset, search_parser, get_unique_filename
from models import MLPClassifier

# --- Constants & Configuration ---
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
GT_CACHE_DIR = os.path.join(CACHE_DIR, "ground_truth")

def ensure_directories():
    """Creates necessary cache directories if they don't exist."""
    os.makedirs(GT_CACHE_DIR, exist_ok=True)

def load_or_compute_ground_truth(X_data, X_query, args, device):
    """
    Computes exact k-NN (Ground Truth) or loads it from cache.
    Returns: true_dists (Tensor), true_indices (Tensor), t_true_avg (float)
    """
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    query_name = os.path.splitext(os.path.basename(args.query))[0]
    
    # Cache filename includes N to ensure we have enough neighbors
    cache_filename = f"gt_{dataset_name}_{query_name}_N{args.N}.npz"
    cache_path = os.path.join(GT_CACHE_DIR, cache_filename)

    if os.path.exists(cache_path):
        print(f"[GroundTruth] Loading cached results from {cache_path}...")
        try:
            data = np.load(cache_path)
            true_dists = torch.from_numpy(data['dists']).to(device)
            true_indices = torch.from_numpy(data['indices']).to(device)
            t_true_avg = float(data['time_avg'])
            print(f"[GroundTruth] Loaded successfully. tTrueAverage: {t_true_avg:.4f} ms")
            return true_dists, true_indices, t_true_avg
        except Exception as e:
            print(f"[GroundTruth] Error loading cache: {e}. Recomputing...")

    print(f"[GroundTruth] Computing exact neighbors for {len(X_query)} queries (N={args.N})...")
    
    t0 = time.time()
    true_indices_list = []
    true_dists_list = []
    batch_size = 100 # Conservative batch size for distance matrix calculation
    
    # Brute force search using PyTorch
    with torch.no_grad():
        for i in range(0, len(X_query), batch_size):
            batch_q = X_query[i : i + batch_size]
            # Euclidean distance: ||a - b||
            dists = torch.cdist(batch_q, X_data)
            # Top-N smallest distances
            vals, inds = torch.topk(dists, args.N, dim=1, largest=False)
            
            true_dists_list.append(vals)
            true_indices_list.append(inds)
            
    true_dists = torch.cat(true_dists_list, dim=0)
    true_indices = torch.cat(true_indices_list, dim=0)
    
    total_time = time.time() - t0
    t_true_avg = (total_time * 1000.0) / len(X_query) # ms per query
    
    print(f"[GroundTruth] Computed in {total_time:.2f}s. Saving to cache...")
    
    # Save to cache
    np.savez(cache_path, 
             dists=true_dists.cpu().numpy(), 
             indices=true_indices.cpu().numpy(),
             time_avg=t_true_avg)
    
    return true_dists, true_indices, t_true_avg

def run_neural_lsh(model, inverted_file, X_data, X_query, args, device):
    """
    Executes the Neural LSH search algorithm.
    Returns: List of result dictionaries, total_approx_time (sec)
    """
    n_query = len(X_query)
    results = []
    total_time = 0.0
    batch_size = 128
    
    print(f"[NeuralLSH] Running search (T={args.T}, N={args.N})...")

    with torch.no_grad():
        for i in range(0, n_query, batch_size):
            end_idx = min(i + batch_size, n_query)
            batch_q = X_query[i:end_idx]
            current_batch_size = end_idx - i
            
            # --- Start Timer ---
            t0_batch = time.time()
            
            # 1. Model Prediction (Forward Pass)
            logits = model(batch_q)
            probs = torch.softmax(logits, dim=1)
            # Get top T bins
            top_bins = torch.topk(probs, args.T, dim=1).indices.cpu().numpy()
            
            # Process each query in the batch
            for j in range(current_batch_size):
                q_idx = i + j
                bins = top_bins[j]
                
                # 2. Candidate Collection
                candidates = []
                for b in bins:
                    if b in inverted_file:
                        candidates.extend(inverted_file[b])
                
                # Remove duplicates
                candidates = list(set(candidates))
                
                res_indices = []
                res_dists = []
                range_neighbors = []
                
                # 3. Exact Search on Candidates
                if len(candidates) > 0:
                    cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
                    cand_vecs = X_data[cand_tensor]
                    q_vec = batch_q[j].unsqueeze(0)
                    
                    # Compute distances to candidates
                    dists_cand = torch.cdist(q_vec, cand_vecs).squeeze(0)
                    
                    # Find nearest N among candidates
                    k_approx = min(args.N, len(candidates))
                    vals, inds = torch.topk(dists_cand, k_approx, largest=False)
                    
                    res_indices = cand_tensor[inds].cpu().tolist()
                    res_dists = vals.cpu().tolist()
                    
                    # 4. Range Search (Optional)
                    if args.range:
                        mask = dists_cand <= args.R
                        range_inds = torch.nonzero(mask).squeeze(1)
                        range_neighbors = cand_tensor[range_inds].cpu().tolist()
                
                results.append({
                    'q_idx': q_idx,
                    'indices': res_indices,
                    'dists': res_dists,
                    'range_neighbors': range_neighbors
                })

            # --- End Timer ---
            total_time += (time.time() - t0_batch)
            
    return results, total_time

def calculate_metrics(results, true_dists, true_indices, n_query, N):
    """Calculates Average AF and Recall@N. Also returns list of AFs for plotting."""
    af_sum = 0.0
    recall_hits = 0
    af_values = []
    
    for res in results:
        q_idx = res['q_idx']
        approx_dists = res['dists']
        approx_indices = res['indices']
        
        # Approximation Factor (AF)
        # AF = dist_approx_1st / dist_true_1st
        true_dist_1 = true_dists[q_idx][0].item()
        
        current_af = 1.0 # Default if perfect
        
        if len(approx_dists) > 0:
            approx_dist_1 = approx_dists[0]
            if true_dist_1 > 1e-9:
                current_af = approx_dist_1 / true_dist_1
            elif approx_dist_1 < 1e-9:
                # Both are effectively 0 -> Perfect
                current_af = 1.0
            else:
                # True is 0 (duplicate exists), Approx is > 0 (missed duplicate)
                # Assign a penalty value to reflect the error
                current_af = 1.0 + (approx_dist_1 * 100.0) 
        
        af_sum += current_af
        af_values.append(current_af)
        
        # Recall@N
        # Intersection of found top N vs true top N
        true_set = set(true_indices[q_idx].cpu().tolist())
        found_set = set(approx_indices)
        recall_hits += len(true_set.intersection(found_set))
        
    avg_af = af_sum / n_query
    recall = recall_hits / (n_query * N)
    
    return avg_af, recall, af_values

def write_output_file(filename, results, metrics, args, true_dists):
    """Writes the results to the output file in the specified format."""
        # Ensure output directory exists
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"[Output] Writing results to {filename}...")
    with open(filename, 'w') as f:
        f.write("METHOD [Neural LSH]\n")
        f.write("===== EVALUATION =====\n")
        f.write(f"Average AF: {metrics['avg_af']:.6f}\n")
        f.write(f"Recall@N: {metrics['recall']:.6f}\n")
        f.write(f"QPS: {metrics['qps']:.2f}\n")
        f.write(f"tApproximateAverage: {metrics['t_approx_avg']:.4f}\n")
        f.write(f"tTrueAverage: {metrics['t_true_avg']:.4f}\n")
        f.write("=============================================================\n")
        
        # Per-Query Results
        for res in results:
            q_idx = res['q_idx']
            f.write(f"Query: {q_idx}\n")
            
            indices = res['indices']
            dists = res['dists']
            
            # Write N neighbors
            for k in range(len(indices)):
                f.write(f"Nearest neighbor-{k+1}: {indices[k]}\n")
                f.write(f"distanceApproximate: {dists[k]:.6f}\n")
                t_dist = true_dists[q_idx][k].item()
                f.write(f"distanceTrue: {t_dist:.6f}\n")
            
            if args.range and res['range_neighbors']:
                f.write("R-near neighbors:\n")
                for rn in res['range_neighbors']:
                    f.write(f"{rn}\n")
            
            f.write("=============================================================\n")

def main():
    # 1. Setup
    ensure_directories()
    args = search_parser()
    
    # Ensure output filename is unique to avoid overwriting previous results
    args.output = get_unique_filename(args.output)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # 2. Load Data
    print(f"[Main] Loading dataset: {args.dataset}")
    X_data_np = get_dataset(args.type, args.dataset)
    print(f"[Main] Loading queries: {args.query}")
    X_query_np = get_dataset(args.type, args.query)
    
    n_data, d_data = X_data_np.shape
    n_query, d_query = X_query_np.shape
    assert d_data == d_query, "Dataset and Query dimensions must match"

    # Move to GPU
    X_data = torch.from_numpy(X_data_np).to(device)
    X_query = torch.from_numpy(X_query_np).to(device)

    # 3. Load Index (Model + Inverted File)
    model_path = f"{args.index_path}_model.pth"
    index_path = f"{args.index_path}_index.pkl"

    if not os.path.exists(model_path) or not os.path.exists(index_path):
        print(f"Error: Index files not found at prefix {args.index_path}")
        print(f"Expected: {model_path} and {index_path}")
        sys.exit(1)

    print(f"[Main] Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint["config"]
    
    model = MLPClassifier(
        d_in=cfg["d_in"],
        n_out=cfg["n_out"],
        hidden_units=cfg["hidden_units"],
        activation=cfg["activation"],
        dropout=cfg["dropout"]
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state"])
    model.eval() # Important: Disable dropout

    print(f"[Main] Loading inverted file from {index_path}...")
    with open(index_path, "rb") as f:
        inverted_file = pickle.load(f)

    # 4. Ground Truth (Exact Search)
    true_dists, true_indices, t_true_avg = load_or_compute_ground_truth(
        X_data, X_query, args, device
    )

    # 5. Run Neural LSH Search
    search_results, total_approx_time = run_neural_lsh(
        model, inverted_file, X_data, X_query, args, device
    )

    # 6. Calculate Metrics
    avg_af, recall, af_values = calculate_metrics(search_results, true_dists, true_indices, n_query, args.N)
    
    metrics = {
        'total_time_ms': total_approx_time * 1000.0,
        'avg_af': avg_af,
        'recall': recall,
        'qps': n_query / total_approx_time if total_approx_time > 0 else 0.0,
        't_approx_avg': (total_approx_time * 1000.0) / n_query,
        't_true_avg': t_true_avg
    }

    # 7. Write Results
    write_output_file(args.output, search_results, metrics, args, true_dists)
    
    # --- PLOT: Approximation Factor Distribution ---
    fig_dir = os.path.join(os.path.dirname(__file__), "fig")
    os.makedirs(fig_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(af_values, bins=50, color='green', edgecolor='black', alpha=0.7)
    plt.xlabel('Approximation Factor (AF)')
    plt.ylabel('Frequency')
    plt.title(f'Approximation Factor Distribution (N={args.N}, T={args.T})')
    plt.grid(axis='y', alpha=0.75)
    
    # Unique filename with params
    range_str = f"_R{args.R}" if args.range else ""
    base_af_name = f"af_dist_{dataset_name}_N{args.N}_T{args.T}{range_str}.png"
    plot_path = get_unique_filename(os.path.join(fig_dir, base_af_name))
    
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved AF distribution plot to {plot_path}")
    # -----------------------------------------------
    
    print("\n[Main] Search Completed Successfully.")
    print(f"  Recall@{args.N}: {recall:.4f}")
    print(f"  Average AF: {avg_af:.4f}")
    print(f"  QPS: {metrics['qps']:.2f}")
    print(f"  Output saved to: {args.output}")

if __name__ == "__main__":
    main()