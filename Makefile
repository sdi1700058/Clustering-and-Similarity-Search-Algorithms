CXX = g++
CXXFLAGS = -O2 -std=c++17 -pthread
BINDIR = bin
TARGET = $(BINDIR)/search

SRC = $(wildcard src/*.cpp) \
	$(wildcard src/utils/*.cpp) \
	$(wildcard src/algorithms/*.cpp)
OBJ = $(patsubst src/%.cpp,$(BINDIR)/%.o,$(SRC))

all: $(TARGET)

search: $(TARGET)
	@ln -sf $(TARGET) $@

$(TARGET): $(OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BINDIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) search
	rm -rf $(BINDIR)
.PHONY: clean all search