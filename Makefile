CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -pthread -Iinclude -MMD -MP -Wall -Wextra -Wpedantic
CPPFLAGS ?=
LDFLAGS ?=
BINDIR ?= bin
TARGET := $(BINDIR)/search

SRC_DIRS := src src/utils src/algorithms src/common
SOURCES := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cpp))
BUILD_DIR := $(BINDIR)/obj
OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPENDS := $(OBJECTS:.o=.d)

all: $(TARGET)

search: $(TARGET)
	@ln -sf $(TARGET) $@

$(TARGET): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

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
	@mkdir -p output
	@i=$$(ls output/hypercube_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/hypercube_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo hypercube -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

run_hypercube_sift: $(TARGET)
	@mkdir -p output
	@i=$$(ls output/hypercube_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/hypercube_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo hypercube -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

check_run_hypercube_mnist:
	@mkdir -p output
	@i=$$(ls output/hypercube_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/hypercube_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo hypercube -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

check_run_hypercube_sift: $(TARGET)
	@mkdir -p output
	@i=$$(ls output/hypercube_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/hypercube_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo hypercube -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

run_lsh_mnist: $(TARGET)
	@mkdir -p output
	@i=$$(ls output/lsh_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/lsh_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo lsh -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

run_lsh_sift: $(TARGET)
	@mkdir -p output
	@i=$$(ls output/lsh_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/lsh_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo lsh -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

check_run_lsh_mnist: $(TARGET)
	@mkdir -p output
	@i=$$(ls output/lsh_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/lsh_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo lsh -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

check_run_lsh_sift: $(TARGET)
	@mkdir -p output
	@i=$$(ls output/lsh_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/lsh_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo lsh -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

run_ivfflat_mnist: $(TARGET)
	@mkdir -p output
	@i=$$(ls output/ivfflat_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfflat_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo ivfflat -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

run_ivfflat_sift: $(TARGET)
	@mkdir -p output
	@i=$$(ls output/ivfflat_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfflat_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	$(TARGET) -algo ivfflat -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

check_run_ivfflat_mnist: $(TARGET)
	@mkdir -p output
	@i=$$(ls output/ivfflat_mnist_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfflat_mnist_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo ivfflat -d data/mnist/train/train-images.idx3-ubyte -q data/mnist/query-test/t10k-images.idx3-ubyte -o $$out -type mnist

check_run_ivfflat_sift: $(TARGET)
	@mkdir -p output
	@i=$$(ls output/ivfflat_sift_*.txt 2>/dev/null \
		| sed -n 's/.*_\([0-9][0-9]*\)\.txt/\1/p' \
		| sort -n \
		| tail -n1); \
	if [ -z "$$i" ]; then i=1; else i=$$((i+1)); fi; \
	out=output/ivfflat_sift_$$i.txt; \
	echo "Running $(TARGET) -> $$out"; \
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(TARGET) -algo ivfflat -d data/sift/sift_base.fvecs -q data/sift/sift_query.fvecs -o $$out -type sift

format:
	clang-format -i $(SOURCES)

clean:
	rm -f $(TARGET) search
	rm -rf $(BUILD_DIR)