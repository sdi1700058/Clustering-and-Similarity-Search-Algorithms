#include <iostream>
#include <unordered_set>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "../../include/algorithms/lsh_search.h"
#include "../../include/utils/args_parser.h"

using namespace std::chrono;

namespace {
static double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
}

std::uint64_t LSHSearch::hash_combine(const std::vector<int>& hvals) const {
    std::uint64_t seed = 1469598103934665603ull; // FNV offset
    const std::uint64_t prime = 1099511628211ull;
    for (int v : hvals) {
        std::uint64_t val = static_cast<std::uint64_t>(v);
        seed ^= val + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        seed *= prime;
    }
    return seed;
}

std::vector<int> LSHSearch::compute_hashes(const HashTable& table, const Vector& v) const {
    std::vector<int> hvals;
    hvals.reserve(table.g.size());
    for (const auto& h : table.g) {
        double projection = dot_product(h.a, v.values);
        double value = (projection + h.b) / p.w;
        int bucket = static_cast<int>(std::floor(value));
        hvals.push_back(bucket);
    }
    return hvals;
}

std::uint64_t LSHSearch::compute_key(const HashTable& table, const Vector& v) const {
    auto hvals = compute_hashes(table, v);
    return hash_combine(hvals);
}

void LSHSearch::build_tables() {
    tables.clear();
    tables.resize(p.L);
    std::uniform_real_distribution<double> dist_b(0.0, p.w);
    std::normal_distribution<double> dist_a(0.0, 1.0);

    for (auto& table : tables) {
        table.g.resize(p.k);
        for (auto& fn : table.g) {
            fn.a.resize(dim);
            for (int d = 0; d < dim; ++d) fn.a[d] = dist_a(rng);
            fn.b = dist_b(rng);
        }
    }

    for (int idx = 0; idx < static_cast<int>(data.size()); ++idx) {
        const auto& vec = data[idx];
        for (auto& table : tables) {
            auto key = compute_key(table, vec);
            table.buckets[key].push_back(idx);
        }
    }
}

void LSHSearch::configure(const Args& args) {
    p.seed = args.seed;
    p.k = args.k;
    p.L = args.L;
    p.w = args.w > 0 ? args.w : default_w;
    p.N = args.N;
    p.R = args.R;
    rng.seed(static_cast<std::mt19937::result_type>(p.seed));
}

void LSHSearch::build_index(const std::vector<Vector>& dataset) {
    data = dataset;
    dim = data.empty() ? 0 : static_cast<int>(data.front().values.size());
    if (dim == 0) {
        std::cout << "[LSH] Empty dataset; nothing to index\n";
        return;
    }
    if (p.L <= 0 || p.k <= 0) {
        throw std::runtime_error("[LSH] Invalid parameters: L and k must be positive");
    }
    build_tables();
    std::cout << "[LSH] Built " << p.L << " hash tables for " << data.size() << " vectors (dim=" << dim << ")\n";
}

SearchResult LSHSearch::search(const Vector& query, const Params& params, int query_id) const {
    auto t0 = high_resolution_clock::now();
    SearchResult result; result.query_id = query_id;
    if (data.empty()) return result;

    std::unordered_set<int> candidates;
    for (const auto& table : tables) {
        auto key = compute_key(table, query);
        auto it = table.buckets.find(key);
        if (it != table.buckets.end()) {
            for (int idx : it->second) candidates.insert(idx);
        }
    }

    if (candidates.empty()) {
        for (size_t i = 0; i < data.size(); ++i) candidates.insert(static_cast<int>(i));
    }

    struct Candidate { int idx; double dist; };
    std::vector<Candidate> scored;
    scored.reserve(candidates.size());
    for (int idx : candidates) {
        double dist = metrics::distance(data[idx].values, query.values, metrics::GLOBAL_METRIC_CFG);
        scored.push_back({idx, dist});
    }
    std::sort(scored.begin(), scored.end(), [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });

    int topK = std::min(params.N, static_cast<int>(scored.size()));
    topK = std::max(topK, 0);
    for (int i = 0; i < topK; ++i) {
        result.neighbor_ids.push_back(scored[i].idx);
        result.distances.push_back(static_cast<float>(scored[i].dist));
    }

    if (params.enable_range && params.R > 0.0) {
        for (const auto& cand : scored) {
            if (cand.dist <= params.R) {
                result.range_neighbor_ids.push_back(cand.idx);
                result.range_distances.push_back(static_cast<float>(cand.dist));
            }
        }
    }

    auto t1 = high_resolution_clock::now();
    result.time_ms = duration<double, std::milli>(t1 - t0).count();
    return result;
}