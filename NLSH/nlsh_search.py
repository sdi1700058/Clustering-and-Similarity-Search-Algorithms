import numpy as np
import torch
import time

# some extra libraries to help
import pickle


# and some utils
from data_parser import get_dataset, search_parser
from models import MLPClassifier
print("hello world")

# ======Parameters======
args = search_parser()

k = args.N # nearest neighbours
T = args.T

batch_size = 128


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =====================





start_time_search = time.time()


# ========Data========
X_train_np = get_dataset(args.type, args.dataset)
X_test_np = get_dataset(args.type, args.query)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train_np).squeeze(1) 

X_test = torch.from_numpy(X_test_np).squeeze(1)

print("\nInsanity check - Data loader")
print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
# ====================


# 1. φορτωσε τα δεδωμενα
start_time = time.time()

with open("inverted_file.pkl", "rb") as f:
    inverted_file = pickle.load(f)

params = torch.load("model.pth")
print("\nInsanity check - model params")
print(f"Number of partitions: {len(inverted_file)}")
print(f"in: {params["config"]['input_size']}")
print(f"hidden: {params["config"]['hidden']}")
print(f"out: {params["config"]['output_size']}")
print(f"activation: {params["config"]['activation_fn']}")
print(f"dp: {params["config"]['dp']}")


model = MLPClassifier(d_in=params["config"]['input_size'], 
                        n_out=params["config"]['output_size'], 
                        hidden_units=params["config"]['hidden'], 
                        activation= params["config"]['activation_fn'], 
                        dropout=params["config"]['dp']).to(device)

model.load_state_dict(params["model_state"])
model.eval()
end_time = time.time()
print(f"\nModel loading completed in {end_time - start_time:.2f} seconds \n")




# 2. Ground truth
print(f"\nComputing ground truth kNN (k={k}) for {X_test.shape[0]} test queries...")
start_time = time.time()

with torch.no_grad():
    dists_all = torch.cdist(X_test, X_train)
    distances_true, indices_true_all = torch.topk(dists_all, k, largest=False, dim=1)
end_time = time.time()
print(f"Ground truth completed in {end_time - start_time:.2f} seconds\n")


# 3. Ξεκηνα την αναζητηση
print("Begin evaluation")
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

            # Get top-k
            _, topk_idx = torch.topk(d, k, largest=False)
            found_neighbors = cand[topk_idx.flatten()]  # [k]

            # Recall@k
            true_set = set(indices_true_all[i+j].tolist())
            found_set = set(found_neighbors.tolist())
            hits = len(true_set.intersection(found_set))

            accuracy_N_count += hits / k
            retrieved_neighbors_count += 1

mean_recall = accuracy_N_count / retrieved_neighbors_count if retrieved_neighbors_count > 0 else 0.0
end_time = time.time()
print(f"Evaluation completed in {(end_time - start_time)/60:.2f} minutes")

print("\n--- Multi-Probe Evaluation Results ---")
print(f"Number of test queries: {retrieved_neighbors_count}")
print(f"Probed bins (T): {T}")
print(f"Target neighbors (k): {k}")
print(f"Mean Recall@{k}: {mean_recall:.4f}")

end_time_search = time.time()
print(f"\nBuild.py completed in {(end_time_search - start_time_search)/60:.2f} minutes \n")

