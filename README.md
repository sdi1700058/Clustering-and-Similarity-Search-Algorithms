# Clustering-and-Similarity-Search-Algorithms
Υλοποίηση αλγορίθμων αναζήτησης διανυσμάτων και συσταδοποίησης σε C++ για d-διάστατους χώρους, βάσει της Euclidean (L2) metric.

Αλγόριθμοι:
1. LSH και Hypercube (random projections).
2. Αναζήτηση με k-means και Inverted File Flat (IVFFlat).
3. Αναζήτηση με k-means και Inverted File Product Quantization (IVFPQ).

Δείτε τη ροή εκτέλεσης στο `src/main.cpp`. Οι επιλογές γραμμής εντολών υλοποιούνται στον parser (`include/utils/args_parser.h`, `src/utils/args_parser.cpp`).

## Build & Run (Makefile)

Για το compiling χρησιμοποιούμε το κατάλληλο Makefile. Ενδεικτικά targets:

```
make run_lsh_mnist
make run_lsh_sift
make run_hypercube_mnist
make run_hypercube_sift
make run_ivfflat_mnist
make run_ivfflat_sift
make run_ivfpq_mnist
make run_ivfpq_sift
```

Στη συνέχεια μπορείτε να ορίσετε παραμέτρους διαδραστικά ή μέσω flags· διαφορετικά χρησιμοποιούνται οι παράμετροι της εκφώνησης. Για καθαρισμό:

```
make clean
```

Υπάρχουν και αντίστοιχα targets για εκτέλεση με Valgrind (`check_run_*algo_*data`), τα οποία απαιτούν σημαντικά περισσότερο χρόνο.

### Common Parameters (CLI)

- `-algo`: lsh | hypercube | ivfflat | ivfpq
- `-d`: dataset file path (π.χ. SIFT: `data/sift/sift_base.fvecs`, MNIST: `data/mnist/train/train-images.idx3-ubyte`)
- `-q`: query file path (π.χ. SIFT: `data/sift/sift_query.fvecs`, MNIST: `data/mnist/query-test/t10k-images.idx3-ubyte`)
- `-o`: output file path (default: `output.txt`)
- `-type`: dataset type (`mnist` ή `sift`)
- `-threads`: number of threads (default: 1)
- `-metric`: l1 | l2 (default: l2)
- `-N`: πλήθος nearest neighbors
- `-R`: ακτίνα για range search
- `-range`: true | false

### CLI Example

```text
=== ANN Framework ===
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
  Seed=1 kclusters=10 nprobe=5
Execution Time (ms): 100274
```

### Output Format Example

```text
=== RESULTS OF IVFPQ ====
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

=============================================================
```

... και αντίστοιχα για όλα τα queries. Η μορφή αρχείου εξόδου παράγεται από τον writer στο `src/utils/result_writer.cpp`.

### Datasets
Οι αλγόριθμοι έχουν δοκιμαστεί με τα εξής datasets:
- MNIST (60k vectors): χρησιμοποιήστε τα αρχεία στον φάκελο `data/mnist`.
- SIFT 1M: είναι μεγάλο για το repository. Κατεβάστε από ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz και κάντε extract στον φάκελο `data/sift/` ή χρησιμοποιήστε το παρακάτω shell command:

```bash
mkdir -p data/sift && wget -P data/sift ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar -xzvf data/sift/sift.tar.gz -C data/sift --strip-components=1 && rm data/sift/sift.tar.gz
```

Σημειώσεις:
- Τα Valgrind runs (`check_run_*`) στο `Makefile` είναι αργά.
- Οι τελικές ρυθμίσεις (config summary) εκτυπώνονται αυτόματα από τον parser.