#!/bin/bash
set -e

echo "=== Building ANN Demo ==="
make clean >/dev/null 2>&1 || true
make -j$(nproc)

echo ""
echo "=== Running Dummy Parallel Search Demo ==="
./bin/search -d data/sample_input.dat -q data/sample_queries.dat -o output/results.txt -type demo -threads 2 -algo dummy
echo ""
echo "=== Output ==="
cat output/results.txt
echo "=== End of Demo ==="


echo "=== Demo: MNIST Loader Test ==="
./bin/search -d data/mnist/train/train-images.idx3-ubyte \
          -q data/mnist/query-test/t10k-images.idx3-ubyte \
          -o output/demo_mnist.txt \
          -algo brute -metric l1 -threads 6 -type mnist
echo ""
echo "=== Output ==="
cat output/demo_mnist.txt
echo "=== End of Demo ==="

echo "=== Demo: SIFT Loader Test ==="
./bin/search -d data/sift/sift_base.fvecs \
          -q data/sift/sift_query.fvecs \
          -o output/demo_sift.txt \
          -algo brute -metric l1 -threads 6 -type sift
echo ""
echo "=== Output ==="
cat output/demo_sift.txt
echo "=== End of Demo ==="

echo "=== Demo: Hypercube Search ==="
./bin/search -d data/mnist/train/train-images.idx3-ubyte \
          -q data/mnist/query-test/t10k-images.idx3-ubyte \
          -o output/demo_hypercube.txt \
          -algo hypercube -metric l2 -threads 6 -type mnist \
          -N 1 -R 2000 -kproj 14 -M 10 -probes 2 -w 4.0 -range true
echo ""
echo "=== Output ==="
cat output/demo_hypercube.txt
echo "=== End of Demo ==="