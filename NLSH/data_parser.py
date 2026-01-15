import numpy as np
import argparse
import os
import struct
import re

def read_idx(filename):
    """Reads MNIST .idx format."""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data

def read_fvecs(filename):
    """Reads SIFT .fvecs format."""
    with open(filename, 'rb') as f:
        # Read the dimension from the first 4 bytes
        d_bytes = f.read(4)
        if not d_bytes:
            return np.array([])
        d = struct.unpack('i', d_bytes)[0]
        
        # Go back to start
        f.seek(0)
        # Calculate number of vectors
        file_size = os.path.getsize(filename)
        n = file_size // (4 * (d + 1))
        
        # Read all data
        data = np.fromfile(f, dtype=np.int32)
        
    if data.size == 0:
        return np.array([])

    # Reshape: n vectors, each with d+1 elements (first is dim)
    data = data.reshape(n, d + 1)
    # Drop the dimension column and view as float32
    vectors = data[:, 1:].view(np.float32)
    return vectors

def get_dataset(DataType, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if DataType == "mnist":
        data = read_idx(file_path).astype(np.float32) / 255.0
        # Flatten images if needed (N, 784)
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        return data

    elif DataType == "sift":
        data = read_fvecs(file_path)
        return data

    elif DataType in ["protein", "generic"]:
        if file_path.endswith('.fvecs'):
            return read_fvecs(file_path)
        # Attempt to load as .npy
        try:
            data = np.load(file_path, allow_pickle=True)
        except Exception:
            # Fallback to fvecs if naming doesn't match but content does? Unlikely.
            # Just raise or retry.
            raise
            
        # Ensure float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        return data

    else:
        raise ValueError(f"Unknown dataset type: {DataType}")


def parse_cpp_output(filename, n_samples, k):
    """
    Parses the output file from the C++ executable to reconstruct the k-NN graph.
    """
    print(f"Parsing C++ output from {filename}...")
    # Initialize with -1 to detect missing neighbors/padding
    indices = np.full((n_samples, k), -1, dtype=int)
    
    nn_pattern = re.compile(r"Nearest neighbor-\d+:\s*(\d+)")
    query_pattern = re.compile(r"Query:\s*(\d+)")
    
    current_query = -1
    neighbor_count = 0
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            q_match = query_pattern.match(line)
            if q_match:
                current_query = int(q_match.group(1))
                neighbor_count = 0
                continue
            
            nn_match = nn_pattern.match(line)
            if nn_match and current_query != -1:
                if neighbor_count < k:
                    neighbor_id = int(nn_match.group(1))
                    indices[current_query, neighbor_count] = neighbor_id
                    neighbor_count += 1
    
    return indices

def save_knn_graph(indices, filename):
    """Saves the k-NN graph indices to a binary file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, indices)
    print(f"k-NN graph saved to {filename}")

def load_knn_graph(filename):
    """Loads the k-NN graph indices from a binary file."""
    if os.path.exists(filename):
        print(f"Loading cached k-NN graph from {filename}...")
        return np.load(filename)
    return None

def get_unique_filename(filepath):
    """
    Checks if a file exists. If it does, appends a number index (_1, _2, ...)
    to the filename until a unique name is found.
    """
    if not os.path.exists(filepath):
        return filepath
    
    base, ext = os.path.splitext(filepath)
    counter = 1
    while True:
        new_filepath = f"{base}_{counter}{ext}"
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1

def build_parser():
    parser = argparse.ArgumentParser(description="NLSH Index Builder")
    # Required
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to dataset file")
    parser.add_argument("-i", "--index", type=str, required=True, help="Path prefix to save index files")
    parser.add_argument("-type", "--type", type=str, default="mnist", help="Dataset type")
    
    # Graph Construction
    parser.add_argument("--knn", type=int, default=10, help="Number of neighbors k for graph")
    parser.add_argument("--graph_method", type=str, default="sklearn", choices=["sklearn", "cpp_file", "cpp_subprocess"], 
                        help="Method to build k-NN graph")
    parser.add_argument("--graph_file", type=str, help="Path to existing graph file (if graph_method=cpp_file)")
    parser.add_argument("--cpp_bin", type=str, default="../bin/search", help="Path to C++ executable")
    parser.add_argument("--cpp_algo", type=str, default="brute", choices=["brute", "lsh", "hypercube", "ivfflat", "ivfpq"], 
                        help="Algorithm to use if running C++ subprocess")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "cosine"], help="Distance metric for graph construction")
    
    # IVFFlat specific arguments for graph construction
    parser.add_argument("--kclusters", type=int, default=50, help="Number of clusters for IVFFlat")
    parser.add_argument("--nprobe", type=int, default=5, help="Number of probes for IVFFlat")

    # KaHIP Options
    parser.add_argument("-m", "--m", type=int, default=100, help="Number of partitions (blocks)")
    parser.add_argument("--imbalance", type=float, default=0.03, help="KaHIP imbalance parameter")
    parser.add_argument("--kahip_mode", type=int, default=2, help="KaHIP mode (0=FAST, 1=ECO, 2=STRONG)")
    
    # MLP Hyperparameters
    parser.add_argument("--layers", type=int, default=3, help="Number of layers in MLP")
    parser.add_argument("--nodes", type=int, default=64, help="Number of nodes per hidden layer")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    
    return parser.parse_args()

def search_parser():
    parser = argparse.ArgumentParser(description="NLSH Search")
    # Required
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to dataset file")
    parser.add_argument("-q", "--query", type=str, required=True, help="Path to query file")
    parser.add_argument("-i", "--index", type=str, required=True, help="Path prefix of the index")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file path")
    parser.add_argument("-type", "--type", type=str, default="mnist", help="Dataset type")
    
    # Search Parameters
    parser.add_argument("-N", type=int, default=1, help="Number of nearest neighbors")
    parser.add_argument("-R", type=float, help="Range search radius")
    parser.add_argument("-T", type=int, default=5, help="Number of probes (bins)")
    parser.add_argument("-range", type=str, default="false", help="Enable range search (true/false)")
    parser.add_argument("-metric", type=str, default="l2", choices=["l2", "cosine"], help="Distance metric")
    
    args = parser.parse_args()
    
    # Handle boolean string for range correctly
    args.range = args.range.lower() in ('true', '1', 'yes')

    # Set default R if not provided
    if args.R is None:
        if args.type == "mnist":
            args.R = 2000.0
        elif args.type == "sift":
            args.R = 2800.0
            
    return args