# Clustering-and-Similarity-Search-Algorithms
Υλοποίηση αλγορίθμων αναζήτησης διανυσμάτων και συσταδοποίησης σε C++.
 - LSH και Hypercube
 - IVFFlat
 - IVFPQ 

Η παρούσα εργασία υλοποιήθηκε στα πλαίσια του μαθήματος «Ανάπτυξη Λογισμικού για Αλ- 
γοριθμικά Προβλήματα» και έχει ως στόχο την ανάπτυξη και την υλοποιήση των ακόλουθων
αλγόριθμων αναζήτησης διανυσμάτων στον d-διάστατο χώρο βάσει της ευκλείδειας μετρικής
(L2):
1. Aλγόριθμο LSH και αλγόριθμο τυχαίας προβολής σε υπερκύβο (Hypercube).
2. Αλγόριθμο αναζήτησης με χρήση συσταδοποίησης k-means και ευρετήριο Inverted
File Flat (IVFFlat).
3. Αλγόριθμο αναζήτησης με χρήση συσταδοποίησης k-means και ευρετήριο Inverted
File Product Quantization (IVFPQ).


Για το compiling των προγραμματων χρεισημοποιούμε κατάληλο Makefile. Πιο συγκεκριμενα: 

    • make run_lsh_mnist
    • make run_lsh_sift
    • make run_hypercube_mnist
    • make run_hypercube_sift
    • make run_ivfflat_mnist
    • make run_ivfflat_sift
    • make run_ivfpq_mnist
    • make run_ivfpq_sift

οπου και στην συνέχεια μπορει ο χρήστης να βάλει τους παραμέτρους που θέλει, αλλιως μπαίνουν οι παράμετροι της εκφώνησης. Ακόμα για τον καθαρισμό των directories χρεισημοποιούμε 
make clean για ολα τα προγράμματα.
Υπάρχουν και αντίστιχοι κανόνες στο Makefile για εκτελέσεις με valgrind (check_run_*algo_*data), οι οποίες όμως παίρνουν πολύ χρόνο για να τρέξουν.

### Common Parameters:
    -algo: lsh, hypercube, ivfflat, ivfpq 
    -d: dataset file path (sift: data/sift/sift_base.fvecs, mnist: data/mnist/train-images.idx3-ubyte)
    -q: query file path (sift: data/sift/sift_query.fvecs, mnist: data/mnist/query-test/t10k-images.idx3-ubyte)
    -o: output file path (default: output.txt)
    -type: dataset type (mnist or sift)
    -threads: number of threads (default: 1)

### CLI Example: 
"=== ANN Framework ===
[INPUT] Enter distance metric (l1/l2) (default: l2) (q to quit): l2
[INPUT] Enter number of threads (default: 1) (q to quit): 10
[INPUT] Enter number of nearest neighbors N (default: 1) (q to quit): 1
[INPUT] Enter search radius R (default: 0.0) (q to quit): 20
[INPUT] Enable range search? (true/false) (default: true) (q to quit): false
[INPUT] Enter seed (default: 1) (q to quit): 1
[INPUT] Enter number of clusters k (default: 50) (q to quit): 10
[INPUT] Enter clusters to probe (default: 5) (q to quit): 5

[INFO] Using configuration:
  Dataset: data/sift/sift_base.fvecs
  Queries: data/sift/sift_query.fvecs
  Output: output/ivfflat_sift_1.txt
  Type: sift
  Algorithm: ivfflat
  Metric: l2
  Threads: 10
  N=1 R=20 Range=false
  Seed=1 kclusters=10 nprobe=5"

### Output Format Example:
"=== RESULTS OF IVFPQ ====
===== CONFIGURATION =====

[INFO] Using configuration:
  Dataset: data/sift/sift_base.fvecs
  Queries: data/sift/sift_query.fvecs
  Output: output/ivfpq_sift_1.txt
  Type: sift
  Algorithm: ivfpq
  Metric: l2
  Threads: 12
  N=2 R=20 Range=false
  Seed=1 kclusters=50 nprobe=5 M=16 nbits=8
Execution Time (ms): 100274
===== EVALUATION =====
Average AF: 1.07447
Recall@N: 0.45245
QPS: 99.7268
tApproximateAverage: 96.8125
tTrueAverage: 177.93
=============================================================
Query: 0
Nearest neighbor-1: 932085
distanceApproximate: 231.535172
distanceTrue: 232.871216
Nearest neighbor-2: 934876
distanceApproximate: 246.943680
distanceTrue: 234.714722

=============================================================
Query: 1
Nearest neighbor-1: 941776
distanceApproximate: 232.211258
distanceTrue: 226.245438
Nearest neighbor-2: 880592
distanceApproximate: 233.229919
distanceTrue: 229.281479

============================================================="
... and so on for all queries

### Datasets:
Η εργασία αυτή δουεύει πάνω σε δυο αρκετά γνωστά Datasets: 
    MNIST 60k vectors
    Sift 1M vectors 
To sift είναι υπερβολικά μεγάλο για να μπεί στο remote repository οπότε και δεν περιλαμβάνεται. Μπορείτε να το κατεβάσετε από εδώ:http://corpus-texmex.irisa.fr/
Να γίνει απλά extract στον φάκελο data/sift/
Για το mnist dataset μπορείτε να χρησιμοποιήσετε τα αρχεία που περιλαμβάνονται στον φάκελο data/mnist.