import numpy as np
from collections import defaultdict

def build_kahip_graph_no_csr(indices, k):
    """
    Constructs a weighted undirected graph from k-NN indices and returns CSR arrays for KaHIP.
    Weights: 2 if mutual neighbor, 1 if one-sided.
    """
    n = indices.shape[0]
    print(f"Converting graph for KaHIP for {n} points with k={k}...")

    # Use a dictionary to count edge occurrences: (u, v) -> count
    edge_counts = defaultdict(int)

    for i in range(n):
        for neighbor in indices[i]:
            if neighbor == -1: continue # Skip padding/invalid indices
            if i == neighbor: continue # Self-loops usually excluded in k-NN but good to check
            
            # Store as sorted tuple to ensure undirectedness
            u, v = (i, neighbor) if i < neighbor else (neighbor, i)
            edge_counts[(u, v)] += 1

    # Build adjacency list
    adj = [[] for _ in range(n)]
    wts = [[] for _ in range(n)]

    for (u, v), count in edge_counts.items():
        # Weight is the count (1 or 2)
        weight = count
        
        adj[u].append(v)
        wts[u].append(weight)
        
        adj[v].append(u)
        wts[v].append(weight)

    # Convert to CSR format arrays
    # xadj: indices where each row starts (size n+1)
    # adjncy: column indices (neighbors)
    # adjcwgt: edge weights
    # vwgt: vertex weights (usually 1)

    xadj = np.zeros(n + 1, dtype=np.int32)
    vwgt = np.ones(n, dtype=np.int32)
    
    # Flatten adj and wts
    all_adjncy = []
    all_adjcwgt = []

    for i in range(n):
        # It is often required/good practice to sort neighbors by index for CSR
        # Zip, sort, unzip
        if adj[i]:
            sorted_pairs = sorted(zip(adj[i], wts[i]), key=lambda x: x[0])
            neighbors, weights = zip(*sorted_pairs)
            all_adjncy.extend(neighbors)
            all_adjcwgt.extend(weights)
        
        xadj[i+1] = len(all_adjncy)

    adjncy = np.array(all_adjncy, dtype=np.int32)
    adjcwgt = np.array(all_adjcwgt, dtype=np.int32)

    print(f"Graph built. Nodes: {n}, Edges: {len(adjncy)//2}")
    
    return xadj, adjncy, adjcwgt, vwgt