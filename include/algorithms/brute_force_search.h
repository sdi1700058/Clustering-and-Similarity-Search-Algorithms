#ifndef BRUTE_FORCE_SEARCH_H
#define BRUTE_FORCE_SEARCH_H

#include "search_algorithm.h"
#include <vector>
#include <list>
#include <utility>
#include <limits>
#include <cmath>
#include <unordered_map>
#include <cassert>

#include "../common/metrics.h"

class BruteForceSearch : public SearchAlgorithm {
private:
    std::vector<Vector> feature_vectors;
    uint32_t n_points = 0;
    uint32_t space_dim = 0;
    static metrics::MetricConfig metric_config;

    static float manhattan_distance(const std::vector<float>& a, const std::vector<float>& b) {
        assert(a.size() == b.size());
        float dist = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) dist += std::fabs(a[i] - b[i]);
        return dist;
    }

public:
    BruteForceSearch() = default;

    static void set_metric_config(const metrics::MetricConfig& cfg);

    void build_index(const std::vector<Vector>& dataset) override;

    SearchResult search(const Vector& query, const Params& params, int query_id) const override;

    // helper: returns ordered list of k nearest (id,dist)
    std::list<std::pair<int,float>> k_nearest(const Vector& query, int k) const;
};

#endif // BRUTE_FORCE_SEARCH_H