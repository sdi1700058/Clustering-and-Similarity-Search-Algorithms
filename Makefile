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
# Protein Search
# ==========================================

PROTEIN_DIR = data/protein
EMBED_SCRIPT = Protein_Search/protein_embed.py
SEARCH_SCRIPT = Protein_Search/protein_search.py
SWISSPROT = $(PROTEIN_DIR)/swissprot_50k.fasta
TARGETS = $(PROTEIN_DIR)/targets.fasta

# 0. Clean old embeddings to avoid confusion
clean_protein_data:
@echo "[Makefile] Cleaning protein embeddings..."
rm -f $(PROTEIN_DIR)/protein_db_*.npy $(PROTEIN_DIR)/protein_db_*.fvecs $(PROTEIN_DIR)/protein_db_*_ids.txt
rm -f $(PROTEIN_DIR)/protein_query_*.npy $(PROTEIN_DIR)/protein_query_*.fvecs $(PROTEIN_DIR)/protein_query_*_ids.txt

# --- Phase 1: L2 Experiments (Unnormalized Vectors) ---

# 1.a Generate Unnormalized Embeddings
protein_data_l2:
	@echo "[Data] Generating UNNORMALIZED protein embeddings for L2..."
	python3 $(P_EMBED) -i $(P_DIR)/swissprot_50k.fasta -o $(P_DIR)/protein_db.npy 
	python3 $(P_EMBED) -i $(P_DIR)/targets.fasta -o $(P_DIR)/targets_vectors.npy

# 1.b Individual L2 Experiments (Cleans Neural Cache before run to force rebuild)
protein_l2_fast: $(TARGET)
	@echo "[Exp] Running L2 FAST..."
	@rm -f $(CACHE_IDX)/protein_index_l2_model.pth $(CACHE_IDX)/protein_index_l2_index.pkl
	python3 $(P_SEARCH) -d $(P_DIR)/protein_db.npy -q $(P_DIR)/targets_vectors.npy \
		-db_fasta $(P_DIR)/swissprot_50k.fasta -q_fasta $(P_DIR)/targets.fasta \
		-config configs/l2_fast_config.json -o output/protein/report_l2_fast.txt

protein_l2_balanced: $(TARGET)
	@echo "[Exp] Running L2 BALANCED..."
	@rm -f $(CACHE_IDX)/protein_index_l2_model.pth $(CACHE_IDX)/protein_index_l2_index.pkl
	python3 $(P_SEARCH) -d $(P_DIR)/protein_db.npy -q $(P_DIR)/targets_vectors.npy \
		-db_fasta $(P_DIR)/swissprot_50k.fasta -q_fasta $(P_DIR)/targets.fasta \
		-config configs/l2_balanced_config.json -o output/protein/report_l2_balanced.txt

protein_l2_accurate: $(TARGET)
	@echo "[Exp] Running L2 ACCURATE..."
	@rm -f $(CACHE_IDX)/protein_index_l2_model.pth $(CACHE_IDX)/protein_index_l2_index.pkl
	python3 $(P_SEARCH) -d $(P_DIR)/protein_db.npy -q $(P_DIR)/targets_vectors.npy \
		-db_fasta $(P_DIR)/swissprot_50k.fasta -q_fasta $(P_DIR)/targets.fasta \
		-config configs/l2_accurate_config.json -o output/protein/report_l2_accurate.txt

protein_l2_extreme: $(TARGET)
	@echo "[Exp] Running L2 EXTREME..."
	@rm -f $(CACHE_IDX)/protein_index_l2_model.pth $(CACHE_IDX)/protein_index_l2_index.pkl
	python3 $(P_SEARCH) -d $(P_DIR)/protein_db.npy -q $(P_DIR)/targets_vectors.npy \
		-db_fasta $(P_DIR)/swissprot_50k.fasta -q_fasta $(P_DIR)/targets.fasta \
		-config configs/l2_extreme_config.json -o output/protein/report_l2_extreme.txt

# 1.c Run All L2 (Sequence: Data -> Fast -> Balanced -> Accurate -> Extreme)
run_protein_l2_all: protein_data_l2
	$(MAKE) protein_l2_fast
	$(MAKE) protein_l2_balanced
	$(MAKE) protein_l2_accurate
	$(MAKE) protein_l2_extreme

# --- Phase 2: Cosine Experiments (Normalized Vectors) ---

# 2.a Generate Normalized Embeddings
protein_data_cosine:
	@echo "[Data] Generating NORMALIZED protein embeddings for Cosine..."
	python3 $(P_EMBED) -i $(P_DIR)/swissprot_50k.fasta -o $(P_DIR)/protein_db.npy --normalize
	python3 $(P_EMBED) -i $(P_DIR)/targets.fasta -o $(P_DIR)/targets_vectors.npy --normalize

# 2.b Individual Cosine Experiments
protein_cosine_fast: $(TARGET)
	@echo "[Exp] Running Cosine FAST..."
	@rm -f $(CACHE_IDX)/protein_index_cosine_model.pth $(CACHE_IDX)/protein_index_cosine_index.pkl
	python3 $(P_SEARCH) -d $(P_DIR)/protein_db.npy -q $(P_DIR)/targets_vectors.npy \
		-db_fasta $(P_DIR)/swissprot_50k.fasta -q_fasta $(P_DIR)/targets.fasta \
		-config configs/cosine_fast_config.json -o output/protein/report_cosine_fast.txt

protein_cosine_balanced: $(TARGET)
	@echo "[Exp] Running Cosine BALANCED..."
	@rm -f $(CACHE_IDX)/protein_index_cosine_model.pth $(CACHE_IDX)/protein_index_cosine_index.pkl
	python3 $(P_SEARCH) -d $(P_DIR)/protein_db.npy -q $(P_DIR)/targets_vectors.npy \
		-db_fasta $(P_DIR)/swissprot_50k.fasta -q_fasta $(P_DIR)/targets.fasta \
		-config configs/cosine_balanced_config.json -o output/protein/report_cosine_balanced.txt

protein_cosine_accurate: $(TARGET)
	@echo "[Exp] Running Cosine ACCURATE..."
	@rm -f $(CACHE_IDX)/protein_index_cosine_model.pth $(CACHE_IDX)/protein_index_cosine_index.pkl
	python3 $(P_SEARCH) -d $(P_DIR)/protein_db.npy -q $(P_DIR)/targets_vectors.npy \
		-db_fasta $(P_DIR)/swissprot_50k.fasta -q_fasta $(P_DIR)/targets.fasta \
		-config configs/cosine_accurate_config.json -o output/protein/report_cosine_accurate.txt

protein_cosine_extreme: $(TARGET)
	@echo "[Exp] Running Cosine EXTREME..."
	@rm -f $(CACHE_IDX)/protein_index_cosine_model.pth $(CACHE_IDX)/protein_index_cosine_index.pkl
	python3 $(P_SEARCH) -d $(P_DIR)/protein_db.npy -q $(P_DIR)/targets_vectors.npy \
		-db_fasta $(P_DIR)/swissprot_50k.fasta -q_fasta $(P_DIR)/targets.fasta \
		-config configs/cosine_extreme_config.json -o output/protein/report_cosine_extreme.txt

# 2.c Run All Cosine (Sequence: Data -> Fast -> Balanced -> Accurate -> Extreme)
run_protein_cosine_all: protein_data_cosine
    $(MAKE) protein_cosine_fast
    $(MAKE) protein_cosine_balanced
    $(MAKE) protein_cosine_accurate
    $(MAKE) protein_cosine_extreme