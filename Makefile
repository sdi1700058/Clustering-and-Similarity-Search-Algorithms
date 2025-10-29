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

.PHONY: clean all search run format

run: $(TARGET)
	$(TARGET) $(RUN_ARGS)

format:
	clang-format -i $(SOURCES)

clean:
	rm -f $(TARGET) search
	rm -rf $(BUILD_DIR)