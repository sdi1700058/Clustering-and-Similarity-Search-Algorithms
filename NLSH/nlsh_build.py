import numpy as np
import torch
import kahip

# some extra libraries to help
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier, NearestNeighbors
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
from scipy.sparse import csr_matrix, triu, tril
from sklearn.datasets import fetch_openml
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle


# and some utils
from data_parser import get_dataset, search_parser, build_parser
from models import MLPClassifier
from graph_utils import adj_matrix_to_csr, knn_to_adj_matrix, build_kahip_graph_no_csr
#print("hello world")

# ======Parameters======
args = build_parser()

k = args.knn # nearest neighbours
rng=args.seed

# KaHIP parameters
imbalance = args.imbalance  
suppress_output = True
mode = args.kahip_mode
num_parts = args.m

# MLP parameters
learning_rate = args.lr
epochs = args.epochs
batch_size = args.batch_size


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 5

# =====================





start_time_build = time.time()

# ========Data========
X_train_np = get_dataset(args.type, args.dataset)
X_test_np = get_dataset(args.type, args.query)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train_np).squeeze(1) 

X_test = torch.from_numpy(X_test_np).squeeze(1)

#print("\nInsanity check - Data loader")
#print(f"Train data shape: {X_train.shape}")
#print(f"Test data shape: {X_test.shape}")
# ====================

# ========Data========
# X_train_np = get_dataset(args.type, args.dataset)

# # Convert to PyTorch tensors
# X_train = torch.from_numpy(X_train_np).squeeze(1) 


#print("\nInsanity check - Data loader")
#print(f"Train data shape: {X_train.shape}")


n_features = 784 #mnist
n_samples = X_train.shape[0] #faster

# ====================


# 1. Κατασκευή Γράφου k-NN
start_time = time.time()

knn = NearestNeighbors(n_neighbors=k+1, algorithm='brute', metric='euclidean') 
knn.fit(X_train_np)
distances, indices = knn.kneighbors(X_train_np) 
#print(distances.shape, indices.shape)
distances, indices = distances[:, 1:], indices[:, 1:]
#print(distances.shape, indices.shape)
end_time = time.time()
#print(f"\nKNN completed in {(end_time - start_time)/60:.2f} minutes \n")


# 2. Προετοιμασία Γράφου - Build CSR arrays  
#print("\nMaking CSR")
start_time = time.time()
xadj, adjncy, adjcwgt, vwgt = build_kahip_graph_no_csr(indices, k)

end_time = time.time()
#print(f"CSR completed in {end_time - start_time:.2f} seconds \n")
#print("\nInsanity check - CSR")
#print("Unique values in xadj:", xadj.shape, np.unique(xadj))
#print("Unique values in adjncy:", adjncy.shape, np.unique(adjncy))
#print("Unique values in adjcwgt:", adjcwgt.shape, np.unique(adjcwgt))
#print("Unique values in vwgt:", vwgt.shape, np.unique(vwgt))




# 3. Ισοκατανεμημένη Διαμέριση (KaHIP) 
#print("Making KaHIP partitions")
start_time = time.time()

edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjcwgt, adjncy,
                               num_parts, imbalance, suppress_output,
                               rng, mode)
end_time = time.time()
#print(f"Partitioning completed in {end_time - start_time:.2f} seconds \n")
#print("\nInsanity check - KaHIP")
#print(f"Edge cut: {edgecut} edges | Avg: {edgecut/n_samples:.4f}")
#print(f"Imbalance tolerance: {imbalance * 100}%")

blocks = torch.tensor(blocks, dtype=torch.long)

#print(blocks.shape)
#print(blocks.min(), blocks.max())  # must be in [0, num_bins-1]
blocks = blocks.to(device)

unique, counts = torch.unique(blocks, return_counts=True)
#print("Unique bins:", len(unique))
#print("Counts:", counts)





train_dataset = TensorDataset(X_train, blocks)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

end_time = time.time()
#print(f"Tests-relabeling completed in {end_time - start_time:.2f} seconds \n")




# 4. Εκπαίδευση Ταξινομητή (PyTorch) 
start_time = time.time()
#print("Begin training")

loss_fn = nn.CrossEntropyLoss()
# hidden = [784, 256]
activation_fn = 'relu'
dp = 0.1
hidden_list = [
    [256],
    [512],
    [784],
    [784, 256],
    [784, 512],
    [784,784],
    [784,784, 512, 256],
    [784,784, 512, 512, 512, 256],
]
for hidden in hidden_list:
    model = MLPClassifier(d_in=n_features, n_out=num_parts, hidden_units=hidden, activation= activation_fn, dropout=dp).to(device)
    opt = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):

        #print("Now in epoch: ", epoch+1)
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward
            preds = model(batch_X)
            loss = loss_fn(preds, batch_y)

            # Backward
            opt.zero_grad()
            loss.backward()
            opt.step()

    end_time = time.time()
    #print(f"Training completed in {end_time - start_time:.2f} seconds \n")


    # #print("Saving model")
    # torch.save({
    #     "model_state": model.state_dict(),
    #     "config": {
    #         "input_size": n_features,
    #         "hidden": hidden,
    #         "output_size": num_parts,
    #         'activation_fn' : activation_fn,
    #         'dp' : dp
    #     }
    # }, "model.pth")

    model.eval()
    inverted_file = defaultdict(list)

    for idx, part_id in enumerate(blocks):
        inverted_file[int(part_id)].append(idx)

    with open("inverted_file.pkl", "wb") as f:
        pickle.dump(inverted_file, f)


    #print("\nInsanity check - Inverted file")
    bin_sizes = np.array([len(inverted_file[b]) for b in sorted(inverted_file.keys())])
    #print("bins:", len(bin_sizes))
    #print("min, median, mean, max:", bin_sizes.min(), np.median(bin_sizes), bin_sizes.mean(), bin_sizes.max())
    small_bins = np.where(bin_sizes < 10)[0]
    #print("bins with <10 items:", small_bins.tolist()[:20])

    end_time_build = time.time()
    print(f"\nBuild completed in {(end_time_build - start_time_build)/60:.2f} minutes \n")




    import numpy as np
    import torch
    import time

    # some extra libraries to help
    import pickle


    # and some utils
    from data_parser import get_dataset, search_parser
    from models import MLPClassifier
    #print("hello world")

    # ======Parameters======



    # =====================





    start_time_search = time.time()









    # 2. Ground truth
    #print(f"\nComputing ground truth kNN (k={k}) for {X_test.shape[0]} test queries...")
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        dists_all = torch.cdist(X_test, X_train)
        distances_true, indices_true_all = torch.topk(dists_all, k, largest=False, dim=1)

    end_time = time.time()
    #print(f"Ground truth completed in {end_time - start_time:.2f} seconds\n")

    # Range search stats
    R = 2800

    # 3. Ξεκηνα την αναζητηση
    #print("Begin evaluation")
    start_time = time.time()

    retrieved_neighbors_count = 0
    accuracy_N_count = 0.0

    with torch.no_grad():
        for i in range(0, X_test.shape[0], batch_size):

            batch_q = X_test[i:i + batch_size]
            B = batch_q.size(0)

            # Compute bin probabilities
            logits = model(batch_q)
            probs = torch.softmax(logits, dim=1)
            top_bins = torch.topk(probs, T, dim=1).indices 

            for j in range(B):

                bins_q = top_bins[j].tolist()

                # Efficient candidate gathering
                cand = []
                for b in bins_q:
                    lst = inverted_file.get(b)
                    if lst is not None:
                        cand += lst

                if not cand:
                    continue

                # Remove duplicates
                cand = torch.tensor(cand, device=device)
                cand = torch.unique(cand)

                if cand.numel() < k:
                    continue

                # Candidate vectors
                Cand_q = X_train[cand]

                # Compute distances
                q = batch_q[j].unsqueeze(0)
                d = torch.cdist(q, Cand_q)

                # ---------- KNN SEARCH ----------
                # Get top-k
                topk_dists, topk_idx = torch.topk(d, k, largest=False)
                found_neighbors = cand[topk_idx.flatten()]  # [k]

                # Recall@k
                true_set = set(indices_true_all[i+j].tolist())
                found_set = set(found_neighbors.tolist())
                hits = len(true_set.intersection(found_set))

                accuracy_N_count += hits / k
                retrieved_neighbors_count += 1

                # ---------- RANGE SEARCH ----------
                ranges_found = cand[(d.squeeze(0) <= R)]
                ranges_dist = d.squeeze(0)[(d.squeeze(0) <= R)]



    mean_recall = accuracy_N_count / retrieved_neighbors_count if retrieved_neighbors_count > 0 else 0.0
    #print(len(ranges_found))
    #print(len(ranges_dist))
    # for some in ranges_found[1:4]:
    #     #print(some)

    # for some in ranges_dist[1:4]:
    #     #print(some)


    end_time = time.time()
    #print(f"\nEvaluation completed in {(end_time - start_time)/60:.2f} minutes")

    print("\n--- Multi-Probe Evaluation Results ---")
    print(f"Number of test queries: {retrieved_neighbors_count}")
    print(f"Probed bins (T): {T}")
    print(f"Target neighbors (k): {k}")
    print(f"Mean Recall@{k}: {mean_recall:.4f}")
    print(hidden)


end_time_search = time.time()
print(f"\nSearch completed in {(end_time_search - start_time_search)/60:.2f} minutes \n")

