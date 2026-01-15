"""
protein_embed.py

Generates protein embeddings using ESM-2 (facebook/esm2_t6_8M_UR50D).
"""
import sys
import os
import argparse
import time
import numpy as np

# --- Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Protein Embeddings")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA")
    parser.add_argument("-o", "--output", required=True, help="Output .npy")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize")
    parser.add_argument("--no_cuda", action="store_true")
    # Add dummy arg for compatibility if Makefile passes it
    parser.add_argument("--model", default=MODEL_NAME, help="Model name (unused, fixed)")
    return parser.parse_args()

def load_model(device_name):
    from transformers import AutoTokenizer, EsmModel
    print(f"[Model] Loading {MODEL_NAME} to {device_name}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.to(device_name)
    model.eval()
    return tokenizer, model

def get_embeddings(sequences, tokenizer, model, device, batch_size=32):
    import torch
    from tqdm import tqdm
    all_embeddings = []
    
    # Filter empty
    valid_seqs = [s for s in sequences if len(s) > 0]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_seqs), batch_size), desc="Embedding"):
            batch = valid_seqs[i:i + batch_size]
            # Truncate to 1022 to allow special tokens (max 1024)
            truncated = [s[:1022] for s in batch]
            
            inputs = tokenizer(truncated, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # Mean pooling over the sequence length (excluding padding is better, but simple mean is standard for ESM)
            # attention_mask: [B, L]
            mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state * mask
            # Sum / Count
            sum_embeddings = token_embeddings.sum(dim=1)
            count = mask.sum(dim=1)
            mean_embeddings = sum_embeddings / count.clamp(min=1)
            
            all_embeddings.append(mean_embeddings.cpu().numpy())

    if all_embeddings:
        return np.vstack(all_embeddings)
    return np.empty((0, 320))

def write_fvecs(filename, data):
    data = np.ascontiguousarray(data, dtype=np.float32)
    n, d = data.shape
    print(f"[IO] Writing C++ compatible {filename} ({n}x{d})...")
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(data[i].tobytes())

def main():
    args = parse_args()
    import torch
    from Bio import SeqIO

    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Read FASTA
    print(f"[Data] Reading {args.input}...")
    ids = []
    sequences = []
    for record in SeqIO.parse(args.input, "fasta"):
        ids.append(str(record.id))
        sequences.append(str(record.seq))

    # 2. Embed
    tokenizer, model = load_model(device)
    embeddings = get_embeddings(sequences, tokenizer, model, device, args.batch_size)

    # 3. Normalize (Crucial for Cosine Similarity via L2/Euclidean indices)
    if args.normalize:
        print("[Preprocessing] L2-normalizing...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

    # 4. Save Outputs
    # A. NPY (For Python/Neural)
    np.save(args.output, embeddings)
    print(f"[IO] Saved .npy to {args.output}")

    # B. IDs (For Mapping Indices -> Names)
    # Handle extensions to create _ids.txt reliably
    base = os.path.splitext(args.output)[0]
    if base.endswith(".dat"): base = base[:-4]
    
    ids_path = f"{base}_ids.txt"
    print(f"[IO] Saving IDs to {ids_path}")
    with open(ids_path, "w") as f:
        for pid in ids:
            f.write(f"{pid}\n")
            
    # C. FVECS (For C++)
    fvecs_path = f"{base}.fvecs"
    write_fvecs(fvecs_path, embeddings)

if __name__ == "__main__":
    main()
