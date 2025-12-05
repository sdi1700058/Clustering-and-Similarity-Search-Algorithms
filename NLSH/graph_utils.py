```import numpy as np
from collections import defaultdict



def build_kahip_graph_no_csr(indices, k):


    n = indices.shape[0]
    print("inside", n)
    # directed neighbor sets for fast lookup
    knn_sets = [set(indices[i]) for i in range(n)]

    # Step 2: adjacency dict for undirected weighted edges
    edge_weights = defaultdict(int)

    for i in range(n):
        for j in knn_sets[i]:
            if i == j:
                continue

            # Undirected key
            a, b = (i, j) if i < j else (j, i)

            if i in knn_sets[j]:
                edge_weights[(a, b)] = 2
            else:
                edge_weights[(a, b)] = 1

    print("edgseweights", len(edge_weights))
    # Step 3: Build adjacency lists for each node
    adj = [[] for _ in range(n)]
    wts = [[] for _ in range(n)]

    for (i, j), w in edge_weights.items():
        adj[i].append(j)
        wts[i].append(w)
        
        adj[j].append(i)
        wts[j].append(w)

    # # Step 4: Sort adjacency lists so CSR is stable
    # for i in range(n):
    #     if len(adj[i]) > 0:
    #         pairs = sorted(zip(adj[i], wts[i]), key=lambda x: x[0])
    #         adj[i] = [p[0] for p in pairs]
    #         wts[i] = [p[1] for p in pairs]

    # Step 5: Convert adjacency lists to CSR arrays
    xadj = np.zeros(n + 1, dtype=np.int32)
    adjncy = np.ones(2*len(edge_weights), dtype=np.int32)
    adjcwgt = np.ones(2*len(edge_weights), dtype=np.int32)
    vwgt = np.ones(n, dtype=np.int32) # 


    total_edges = 0
    pos = 0
    for i in range(n):
        xadj[i+1] = xadj[i] + len(adj[i])

        deg = len(adj[i])
        adjncy[pos:pos+deg] = adj[i]
        adjcwgt[pos:pos+deg] = wts[i]
        pos += deg

    print("total edgdes", total_edges)

    print(xadj.shape)
    print(adjncy.shape)
    print(adjcwgt.shape)

    return xadj, adjncy, adjcwgt, vwgt




import numpy as np

def knn_to_adj_matrix(indices):

    n = indices.shape[0]
    adj_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in indices[i]:
            if i != j:   
                adj_matrix[i, j] += 1
                adj_matrix[j, i] += 1

    return adj_matrix




def adj_matrix_to_csr(indices):

    n = indices.shape[0]
    xadj = np.zeros(n + 1, dtype=np.int32)
    vwgt = np.ones(n, dtype=np.int32) # 

    adjncy = []
    adjcwgt = []

    for i in range(n):
        neighbors = np.nonzero(adj_matrix[i])[0]  
        weights = adj_matrix[i, neighbors]

        sorted_idx = np.argsort(neighbors)
        neighbors = neighbors[sorted_idx]
        weights = weights[sorted_idx]

        adjncy.extend(neighbors)
        adjcwgt.extend(weights)

        xadj[i+1] = xadj[i] + len(neighbors)

    adjncy = np.array(adjncy, dtype=np.int32)
    adjcwgt = np.array(adjcwgt, dtype=np.float32)


    return xadj, adjncy, adjcwgt, 







def myCSR(indices):
    print("\nMaking CSR")
    start_time = time.time()

    xadj = [0] # cumulative index # 
    adjncy = [] # neighbor indices # 
    # adjcwgt = [] # edge weights (distances) #
    adjcwgt = np.zeros(n_samples, dtype=np.int32) # 
    vwgt = np.ones(n_samples, dtype=np.int32) # 
    for i in range(n_samples): # 

        adjncy.extend(neighbors) # 
        xadj.append(len(adjncy)) # 

        for j in indices[i]:
            if i in indices[j]:
                adjcwgt.appene(2)
            else:
                adjcwgt.appene(1)

    xadj = np.array(xadj, dtype=np.int32) # 
    adjncy = np.array(adjncy, dtype=np.int32) # 
    adjcwgt = np.array(adjcwgt, dtype=np.float32) # distances as floats "
    end_time = time.time()
    print(f"CSR completed in {end_time - start_time:.2f} seconds \n")



```