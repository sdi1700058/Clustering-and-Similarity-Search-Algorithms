"""
protein_embed.py

Generates protein embeddings using ESM-2 (facebook/esm2_t6_8M_UR50D).
"""
import sys
import os
import pickle
import argparse
import json
import time
import numpy as np
# Deferred imports to prevent crashes during --help or unexpected OOMs
# import torch
# from Bio import SeqIO
# from transformers import AutoTokenizer, EsmModel
# from tqdm import tqdm

# --- Configuration & Imports ---
# Add parent directory to path to import from NLSH for utils
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

try:
    from NLSH.data_parser import get_unique_filename
except ImportError:
    def get_unique_filename(filepath):
        if not os.path.exists(filepath): return filepath
        base, ext = os.path.splitext(filepath)
        idx = 1
        while os.path.exists(f"{base}_{idx}{ext}"): idx += 1
        return f"{base}_{idx}{ext}"

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Protein Embeddings using ESM-2")
    parser.add_argument("-i", "--input", required=True, help="Path to input FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Path to output .dat file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings for cosine similarity")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU usage")
    return parser.parse_args()

def load_model(device_name):
    from transformers import AutoTokenizer, EsmModel

    print(f"[Model] Loading {MODEL_NAME} to {device_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = EsmModel.from_pretrained(MODEL_NAME)
    except OSError:
        print(f"Error: Could not load model '{MODEL_NAME}'. Check internet connection.")
        sys.exit(1)
    model.to(device_name)
    model.eval()
    return tokenizer, model

def get_embeddings(sequences, tokenizer, model, device, batch_size=32):
    import torch
    from tqdm import tqdm

    all_embeddings = []
    
    # Filter out empty sequences to avoid errors
    valid_sequences = [s for s in sequences if len(s) > 0]
    if len(valid_sequences) < len(sequences):
        print(f"[Warning] Skipped {len(sequences) - len(valid_sequences)} empty sequences.")

    total_batches = (len(valid_sequences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(valid_sequences), batch_size), desc="Embedding", total=total_batches):
            batch_seqs = valid_sequences[i : i + batch_size]
            
            # Tokenize
            try:
                # Truncating at 1024 is still heavy for 8GB VRAM with ESM-2
                # Reduced max_length to 512 for safety on smaller GPUs
                inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                
                # --- CORRECT Mean Pooling strategy ---
                token_embeddings = outputs.last_hidden_state # [Batch, SeqLen, 320]
                
                # Attention mask: [Batch, SeqLen] -> [Batch, SeqLen, 1]
                input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                
                # Sum embeddings ignoring padding
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                
                # Count tokens (avoid div by zero)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Mean
                batch_embeddings = sum_embeddings / sum_mask
                
                all_embeddings.append(batch_embeddings.cpu().numpy())

                # Explicit cleanup
                del inputs, outputs, token_embeddings, sum_embeddings, batch_embeddings
                torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n[Error] OOM at batch {i}. Try reducing --batch_size further.")
                    torch.cuda.empty_cache()
                    sys.exit(1)
                else:
                    raise e
            
    if all_embeddings:
        return np.vstack(all_embeddings)
def main():
    args = parse_args()
    
    import torch
    from Bio import SeqIO

    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Read FASTA
    print(f"[Data] Reading sequences from {args.input}...")
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        sys.exit(1)

    ids = []
    sequences = []
    for record in SeqIO.parse(args.input, "fasta"):
        ids.append(str(record.id))
        sequences.append(str(record.seq))
        
    print(f"[Data] Parsed {len(sequences)} sequences.")

    # 2. Load Model
    tokenizer, model = load_model(device)

    # 3. Generate Embeddings
    print(f"[Inference] Generating embeddings...")
    embeddings = get_embeddings(sequences, tokenizer, model, device, args.batch_size)

    # NORMALIZE FOR COSINE SIMILARITY COMPATIBILITY (Essential for ESM-2)
    if args.normalize:
        print(f"[Preprocessing] L2-normalizing embeddings...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)
        suffix = "_norm"
    else:
        print(f"[Preprocessing] Skipping normalization (Raw Embeddings).")
        suffix = "_raw"

    print(f"[Result] Generated embeddings shape: {embeddings.shape}")

    # 4. Save Output
    # Construct filename: base + suffix + .npy
    base_name_no_ext = os.path.splitext(args.output)[0]
    candidate_filename = f"{base_name_no_ext}{suffix}.npy"
    final_output_path = candidate_filename
    
    print(f"[IO] Saving vectors to {final_output_path}...")
    with open(final_output_path, "wb") as f:
        np.save(f, embeddings)

    # Save Metadata (Sidecar JSON)
    metadata_path = os.path.splitext(final_output_path)[0] + ".json"
    metadata = {
        "model": MODEL_NAME,
        "normalized": args.normalize,
        "n_samples": embeddings.shape[0],
        "n_features": embeddings.shape[1],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_fasta": args.input
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[IO] Saved metadata to {metadata_path}")

    # Save IDs
    # Derive IDs filename from the unique output filename to avoid collisions
    # e.g., "db_norm_1.npy" -> "db_norm_1_ids.txt"
    ids_path = os.path.splitext(final_output_path)[0] + "_ids.txt"
        
    print(f"[IO] Saving IDs to {ids_path}...")
    with open(ids_path, "w") as f:
        for pid in ids:
            f.write(f"{pid}\n")
            
    print("Done.")

if __name__ == "__main__":
    main()
