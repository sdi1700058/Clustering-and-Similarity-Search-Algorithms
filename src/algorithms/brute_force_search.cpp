#include <chrono>
#include <iostream>
#include <algorithm>
#include <limits>

#include "../../include/algorithms/brute_force_search.h"
#include "../../include/common/metrics.h"

metrics::MetricConfig BruteForceSearch::metric_config{};

void BruteForceSearch::set_metric_config(const metrics::MetricConfig& cfg) {
    metric_config = cfg;
}

void BruteForceSearch::build_index(const std::vector<Vector>& dataset) {
    feature_vectors = dataset;
    n_points = static_cast<uint32_t>(dataset.size());
    if (n_points > 0) space_dim = static_cast<uint32_t>(dataset[0].values.size());
    std::cout << "[BruteForce] Index built with " << n_points << " points (dim=" << space_dim << ").\n";
}

SearchResult BruteForceSearch::search(const Vector& query, const Params& params, int query_id) const {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (feature_vectors.empty()) {
        std::cerr << "[BruteForce] ERROR: empty dataset\n";
        return SearchResult{};
    }

    struct Pair { int id; double dist; };
    std::vector<Pair> all;
    all.reserve(n_points);

    for (uint32_t i = 0; i < n_points; ++i) {
    double d = metrics::distance(feature_vectors[i].values, query.values, metric_config);
        all.push_back({static_cast<int>(i), d});
    }

    int k = params.N > 0 ? params.N : 1;
    if (k > (int)all.size()) k = (int)all.size();

    std::partial_sort(all.begin(), all.begin() + k, all.end(),
                      [](const Pair& a, const Pair& b){ return a.dist < b.dist; });

    SearchResult res;
    res.query_id = query_id;
    for (int i = 0; i < k; ++i) {
        res.neighbor_ids.push_back(all[i].id);
        res.distances.push_back(static_cast<float>(all[i].dist));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

std::list<std::pair<int,float>> BruteForceSearch::k_nearest(const Vector& query, int k) const {
    std::list<std::pair<int,float>> out;
    if (feature_vectors.empty()) return out;
    std::vector<std::pair<int,float>> all;
    all.reserve(n_points);
    for (uint32_t i = 0; i < n_points; ++i)
    all.emplace_back(i, metrics::distance(feature_vectors[i].values, query.values, metric_config));
    if (k > (int)all.size()) k = (int)all.size();
    std::partial_sort(all.begin(), all.begin() + k, all.end(), [](auto&a, auto&b){ return a.second < b.second; });
    for (int i = 0; i < k; ++i) out.push_back(all[i]);
    return out;
}