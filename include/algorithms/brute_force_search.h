#ifndef BRUTE_FORCE_SEARCH_H
#define BRUTE_FORCE_SEARCH_H

#include <vector>
#include <list>
#include <utility>
#include <limits>
#include <cmath>
#include <unordered_map>
#include <cassert>

#include "search_algorithm.h"
#include "../common/metrics.h"

class BruteForceSearch : public SearchAlgorithm {
private:
    std::vector<Vector> feature_vectors;
    uint32_t n_points = 0;
    uint32_t space_dim = 0;
public:
    BruteForceSearch() = default;
    void build_index(const std::vector<Vector>& dataset) override;
    SearchResult search(const Vector& query, const Params& params, int query_id) const override;
    std::list<std::pair<int,float>> k_nearest(const Vector& query, int k) const;
    void configure(const Args& args) override { (void)args; } // brute uses global defaults
    std::string name() const override { return "BruteForce"; }
};

#endif // BRUTE_FORCE_SEARCH_H