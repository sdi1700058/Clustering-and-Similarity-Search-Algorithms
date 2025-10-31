#include <chrono>
#include <iostream>
#include <algorithm>
#include <limits>
#include <list>
#include <utility>
#include <vector>
#include <cmath>

#include "../../include/algorithms/brute_force_search.h"
#include "../../include/utils/args_parser.h"
#include "../../include/common/metrics.h"

void BruteForceSearch::build_index(const std::vector<Vector>& dataset) {
    feature_vectors = dataset;
    n_points = static_cast<uint32_t>(dataset.size());
    space_dim = (n_points>0) ? static_cast<uint32_t>(dataset[0].values.size()) : 0;
    std::cout << "[BruteForce] built index with " << n_points << " points (dim=" << space_dim << ")\n";
}

SearchResult BruteForceSearch::search(const Vector& query, const Params& params, int query_id) const {
    auto t0 = std::chrono::high_resolution_clock::now();
    SearchResult res; res.query_id = query_id;
    if (n_points == 0) return res;

    const int k_requested = params.N > 0 ? params.N : 1;
    const int k = std::min(k_requested, static_cast<int>(n_points));

    std::vector<std::pair<int, double>> best;
    best.reserve(k);
    double worst_dist = std::numeric_limits<double>::infinity();
    std::size_t worst_pos = 0;
    auto recompute_worst = [&]() {
        if (best.empty()) {
            worst_dist = std::numeric_limits<double>::infinity();
            return;
        }
        auto it = std::max_element(best.begin(), best.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; });
        worst_pos = static_cast<std::size_t>(std::distance(best.begin(), it));
        worst_dist = it->second;
    };
    auto insert_candidate = [&](int idx, double dist) {
        if (static_cast<int>(best.size()) < k) {
            best.emplace_back(idx, dist);
            recompute_worst();
        } else if (dist < worst_dist) {
            best[worst_pos] = {idx, dist};
            recompute_worst();
        }
    };

    for (uint32_t i = 0; i < n_points; ++i) {
        double dist = metrics::distance(feature_vectors[i].values, query.values, metrics::GLOBAL_METRIC_CFG);

        if (k > 0 && (static_cast<int>(best.size()) < k || dist < worst_dist)) {
            insert_candidate(static_cast<int>(i), dist);
        }

        if (params.enable_range && params.R > 0.0 && dist <= params.R) {
            res.range_neighbor_ids.push_back(static_cast<int>(i));
            res.range_distances.push_back(static_cast<float>(dist));
        }
    }

    std::sort(best.begin(), best.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
    for (const auto& item : best) {
        res.neighbor_ids.push_back(item.first);
        res.distances.push_back(static_cast<float>(item.second));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

std::list<std::pair<int,float>> BruteForceSearch::k_nearest(const Vector& query, int k) const {
    std::list<std::pair<int,float>> out;
    if (n_points == 0 || k <= 0) return out;

    std::vector<std::pair<int, double>> best;
    best.reserve(k);
    double worst_dist = std::numeric_limits<double>::infinity();
    std::size_t worst_pos = 0;
    auto recompute_worst = [&]() {
        if (best.empty()) {
            worst_dist = std::numeric_limits<double>::infinity();
            return;
        }
        auto it = std::max_element(best.begin(), best.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; });
        worst_pos = static_cast<std::size_t>(std::distance(best.begin(), it));
        worst_dist = it->second;
    };
    auto insert_candidate = [&](int idx, double dist) {
        if (static_cast<int>(best.size()) < k) {
            best.emplace_back(idx, dist);
            recompute_worst();
        } else if (dist < worst_dist) {
            best[worst_pos] = {idx, dist};
            recompute_worst();
        }
    };

    for (uint32_t i = 0; i < n_points; ++i) {
        double dist = metrics::distance(feature_vectors[i].values, query.values, metrics::GLOBAL_METRIC_CFG);
        if (static_cast<int>(best.size()) < k || dist < worst_dist) {
            insert_candidate(static_cast<int>(i), dist);
        }
    }

    for (const auto& item : best) {
        out.emplace_back(item.first, static_cast<float>(item.second));
    }
    return out;
}