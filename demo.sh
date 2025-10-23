#!/bin/bash
set -e

echo "=== Building ANN Demo ==="
make clean >/dev/null 2>&1 || true
make -j$(nproc)

echo ""
echo "=== Running Dummy Parallel Search Demo ==="
./bin/search -d data/sample_input.dat -q data/sample_queries.dat -o output/results.txt -type demo -threads 4

echo ""
echo "=== Output ==="
cat output/results.txt
echo "=== End of Demo ==="