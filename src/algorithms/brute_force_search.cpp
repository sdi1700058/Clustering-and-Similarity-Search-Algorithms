#include <chrono>
#include <iostream>
#include <algorithm>
#include <queue>
#include <limits>
#include <list>
#include <utility>
#include <unordered_set>

#include "../../include/algorithms/brute_force_search.h"
#include "../../include/utils/args_parser.h"
#include "../../include/common/metrics.h"

using namespace std::chrono;

void BruteForceSearch::build_index(const std::vector<Vector>& dataset) {
    feature_vectors = dataset;
    n_points = dataset.empty() ? 0 : static_cast<int>(dataset.size());
    space_dim = dataset.empty() ? 0 : static_cast<int>(dataset.front().values.size());
    std::cout << "[BruteForce] built index with " << n_points << " points (dim=" << space_dim << ")\n";
}

SearchResult BruteForceSearch::search(const Vector& query, const Params& params, int query_id) const {
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();

    SearchResult res;
    res.query_id = query_id;
    if (n_points == 0) return res;

    const int N = params.N;
    const bool do_range = params.enable_range && params.R > 0.0;

    // Use a fixed-size max-heap to keep top-N smallest distances
    std::priority_queue<std::pair<double, int>> topN; // (distance, id)

    for (uint32_t i = 0; i < n_points; ++i) {
        double dist = metrics::distance(
            query.values,
            feature_vectors[i].values,
            metrics::GLOBAL_METRIC_CFG
        );

        if ((int)topN.size() < N) {
            topN.emplace(dist, i);
        } else if (dist < topN.top().first) {
            topN.pop();
            topN.emplace(dist, i);
        }

        if (do_range && dist <= params.R) {
            res.range_neighbor_ids.push_back(i);
            res.range_distances.push_back(static_cast<float>(dist));
        }
    }

    // Extract and sort final results
    res.neighbor_ids.resize(topN.size());
    res.distances.resize(topN.size());
    for (int i = (int)topN.size() - 1; i >= 0; --i) {
        res.neighbor_ids[i] = topN.top().second;
        res.distances[i] = static_cast<float>(topN.top().first);
        topN.pop();
    }

    auto t1 = high_resolution_clock::now();
    res.time_ms = duration<double, std::milli>(t1 - t0).count();
    return res;
}