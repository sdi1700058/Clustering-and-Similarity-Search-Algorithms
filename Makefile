CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -pthread -fPIE -IANN/include -MMD -MP -Wall -Wextra -Wpedantic
CPPFLAGS ?=
LDFLAGS ?=
BINDIR ?= bin
TARGET := $(BINDIR)/search

SRC_DIRS := ANN/src ANN/src/utils ANN/src/algorithms ANN/src/common
SOURCES := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cpp))
BUILD_DIR := $(BINDIR)/obj
OBJECTS := $(patsubst ANN/src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPENDS := $(OBJECTS:.o=.d)

all: $(TARGET)

search: $(TARGET)
	@ln -sf $(TARGET) $@

$(TARGET): $(OBJECTS)
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: ANN/src/%.cpp
	@mkdir -p $(dir $@)
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEPENDS)

.PHONY: clean all search run format run_hypercube_mnist run_hypercube_sift \
	check_run_hypercube_mnist check_run_hypercube_sift \
	run_lsh_mnist run_lsh_sift \
	check_run_lsh_mnist check_run_lsh_sift \
	run_ivfflat_mnist run_ivfflat_sift \
	check_run_ivfflat_mnist check_run_ivfflat_sift

run: $(TARGET)
	$(TARGET) $(RUN_ARGS)

run_hypercube_mnist: $(TARGET)
	@mkdir -p output/hypercube
	@i=$$(ls output/hypercube/hypercube_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/hypercube/hypercube_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo hypercube -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

run_hypercube_sift: $(TARGET)
	@mkdir -p output/hypercube
	@i=$$(ls output/hypercube/hypercube_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/hypercube/hypercube_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo hypercube -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

check_run_hypercube_mnist:
	@mkdir -p output/hypercube
	@i=$$(ls output/hypercube/hypercube_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/hypercube/hypercube_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo hypercube -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

check_run_hypercube_sift: $(TARGET)
	@mkdir -p output/hypercube
	@i=$$(ls output/hypercube/hypercube_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/hypercube/hypercube_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo hypercube -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

run_lsh_mnist: $(TARGET)
	@mkdir -p output/lsh
	@i=$$(ls output/lsh/lsh_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/lsh/lsh_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo lsh -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

run_lsh_sift: $(TARGET)
	@mkdir -p output/lsh
	@i=$$(ls output/lsh/lsh_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/lsh/lsh_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo lsh -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

check_run_lsh_mnist: $(TARGET)
	@mkdir -p output/lsh
	@i=$$(ls output/lsh/lsh_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/lsh/lsh_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo lsh -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

check_run_lsh_sift: $(TARGET)
	@mkdir -p output/lsh
	@i=$$(ls output/lsh/lsh_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/lsh/lsh_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo lsh -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

run_ivfflat_mnist: $(TARGET)
	@mkdir -p output/ivfflat
	@i=$$(ls output/ivfflat/ivfflat_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfflat/ivfflat_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo ivfflat -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

run_ivfflat_sift: $(TARGET)
	@mkdir -p output/ivfflat
	@i=$$(ls output/ivfflat/ivfflat_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfflat/ivfflat_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo ivfflat -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

check_run_ivfflat_mnist: $(TARGET)
	@mkdir -p output/ivfflat
	@i=$$(ls output/ivfflat/ivfflat_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfflat/ivfflat_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo ivfflat -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

check_run_ivfflat_sift: $(TARGET)
	@mkdir -p output/ivfflat
	@i=$$(ls output/ivfflat/ivfflat_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfflat/ivfflat_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo ivfflat -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

run_ivfpq_mnist: $(TARGET)
	@mkdir -p output/ivfpq
	@i=$$(ls output/ivfpq/ivfpq_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfpq/ivfpq_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo ivfpq -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

run_ivfpq_sift: $(TARGET)
	@mkdir -p output/ivfpq
	@i=$$(ls output/ivfpq/ivfpq_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfpq/ivfpq_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo ivfpq -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

check_run_ivfpq_mnist: $(TARGET)
	@mkdir -p output/ivfpq
	@i=$$(ls output/ivfpq/ivfpq_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfpq/ivfpq_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo ivfpq -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

check_run_ivfpq_sift: $(TARGET)
	@mkdir -p output/ivfpq
	@i=$$(ls output/ivfpq/ivfpq_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfpq/ivfpq_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo ivfpq -d data/sift/sift_base

run_brute_mnist: $(TARGET)
	@mkdir -p output/brute
	@i=$$(ls output/brute/brute_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/brute/brute_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo brute -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

run_brute_sift: $(TARGET)
	@mkdir -p output/brute
	@i=$$(ls output/brute/brute_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/brute/brute_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo brute -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

format:
	clang-format -i $(SOURCES)

clean:
	rm -f $(TARGET) search
	rm -rf $(BUILD_DIR)

# ==========================================
# Protein Search Tasks
# ==========================================

PROTEIN_DIR = data/protein
EMBED_SCRIPT = Protein_Search/protein_embed.py
SEARCH_SCRIPT = Protein_Search/protein_search.py

# 1. Generate Embeddings (Run Once) - Normalized for Cosine Similarity
protein_embed:
	@echo "[Makefile] Generating Database Embeddings (this may take a while)..."
	python3 $(EMBED_SCRIPT) -i $(PROTEIN_DIR)/swissprot_50k.fasta -o $(PROTEIN_DIR)/protein_db.dat --normalize --batch_size 16
	@echo "[Makefile] Generating Query Embeddings..."
	python3 $(EMBED_SCRIPT) -i $(PROTEIN_DIR)/targets.fasta -o $(PROTEIN_DIR)/targets_vectors.dat --normalize --batch_size 16

# 2. Run Full Search Benchmark
protein_search: $(TARGET)
	@echo "[Makefile] Running Protein Search Pipeline..."
	@mkdir -p output/protein
	python3 $(SEARCH_SCRIPT) \
		-d $(PROTEIN_DIR)/protein_db.dat \
		-q $(PROTEIN_DIR)/targets_vectors.dat \
		-db_fasta $(PROTEIN_DIR)/swissprot_50k.fasta \
		-q_fasta $(PROTEIN_DIR)/targets.fasta \
		-o output/protein/final_report.txt \
		-config configs/protein_cosine.json \
		-method all \
		-N 50