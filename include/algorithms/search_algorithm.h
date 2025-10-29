#ifndef SEARCH_ALGORITHM_H
#define SEARCH_ALGORITHM_H

/*
This file defines the abstract base class for search algorithms.
All search algorithms should inherit from this class and implement
the build_index and search methods.
*/

#include <vector>
#include <string>

// Basic vector & results
struct Vector {
    std::vector<double> values;
};

struct SearchResult {
    int query_id = -1;
    std::vector<int> neighbor_ids;
    std::vector<float> distances;
    std::vector<int> range_neighbor_ids;
    std::vector<float> range_distances;
    double time_ms = 0.0;
};

struct Params {
    int N = 1;
    double R = 0.0;
    bool enable_range = false;
};

// Args container forward-declared to allow configure
struct Args;

// Common algorithm interface (all algorithms must implement)
class SearchAlgorithm {
public:
    virtual ~SearchAlgorithm() = default;
    // build index (from dataset)
    virtual void build_index(const std::vector<Vector>& dataset) = 0;
    // run a single query
    virtual SearchResult search(const Vector& query, const Params& params, int query_id) const = 0;
    // configure algorithm with CLI args (defaults set by parse)
    virtual void configure(const Args& args) { (void)args; }
    // name for output header
    virtual std::string name() const = 0;
};

#endif // SEARCH_ALGORITHM_H