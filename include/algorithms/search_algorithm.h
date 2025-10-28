#ifndef SEARCH_ALGORITHM_H
#define SEARCH_ALGORITHM_H

/*
This file defines the abstract base class for search algorithms.
All search algorithms should inherit from this class and implement
the build_index and search methods.
*/

#include <vector>
#include <string>

// Simple vector representation
struct Vector {
    std::vector<float> values;
};

// Structure to hold search results
struct SearchResult {
    int query_id;
    std::vector<int> neighbor_ids;
    std::vector<float> distances;
    double time_ms;
};

// Parameters for search algorithms
struct Params {
    int N = 5;
    double R = 100.0;
};

// Abstract base class for search algorithms
class SearchAlgorithm {
public:
    virtual void build_index(const std::vector<Vector>& dataset) = 0;
    virtual SearchResult search(const Vector& query, const Params& params, int query_id) const = 0;
    virtual ~SearchAlgorithm() {}
};

#endif // SEARCH_ALGORITHM_H