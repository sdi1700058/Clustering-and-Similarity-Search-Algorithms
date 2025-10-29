#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "../../include/algorithms/hypercube_search.h"
#include "../../include/utils/args_parser.h"
#include "../../include/common/metrics.h"

namespace {
using Clock = std::chrono::high_resolution_clock;
}

void HypercubeSearch::configure(const Args& args) {
    seed_ = args.seed;
    kproj_ = std::max(1, args.kproj);
    if (kproj_ > 32) {
        std::cerr << "[Hypercube] capping kproj from " << kproj_ << " to 32\n";
        kproj_ = 32;
    }
    max_candidates_ = std::max(0, args.M);
    max_probes_ = std::max(1, args.probes);
    w_ = args.w > 0.0 ? args.w : 4.0;
}

void HypercubeSearch::build_index(const std::vector<Vector>& dataset) {
    dataset_ = dataset;
    cube_.clear();
    projections_.clear();
    offsets_.clear();
    space_dim_ = dataset_.empty() ? 0u : static_cast<uint32_t>(dataset_[0].values.size());

    if (dataset_.empty()) {
        std::cout << "[Hypercube] dataset is empty, index cleared\n";
        return;
    }
    if (space_dim_ == 0) {
        throw std::runtime_error("[Hypercube] dataset vectors have zero dimension");
    }
    if (static_cast<size_t>(kproj_) > sizeof(uint32_t) * 8u) {
        throw std::runtime_error("[Hypercube] kproj cannot exceed 32");
    }
    if (w_ <= 0.0) {
        throw std::runtime_error("[Hypercube] window size w must be positive");
    }

    std::mt19937 rng(static_cast<uint32_t>(seed_));
    std::normal_distribution<double> gaussian(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, w_);

    projections_.assign(kproj_, std::vector<double>(space_dim_));
    offsets_.resize(kproj_);

    for (int i = 0; i < kproj_; ++i) {
        for (uint32_t d = 0; d < space_dim_; ++d) {
            projections_[i][d] = gaussian(rng);
        }
        offsets_[i] = uniform(rng);
    }

    cube_.reserve(dataset_.size());
    for (size_t idx = 0; idx < dataset_.size(); ++idx) {
        if (dataset_[idx].values.size() != space_dim_) {
            throw std::runtime_error("[Hypercube] inconsistent vector dimensionality");
        }
        uint32_t bucket = hash_vector(dataset_[idx].values);
        cube_[bucket].push_back(static_cast<int>(idx));
    }

    std::cout << "[Hypercube] built index with " << dataset_.size()
              << " points (dim=" << space_dim_ << ")\n";
    if (metrics::GLOBAL_METRIC_CFG.type != metrics::MetricType::L2) {
        std::cerr << "[Hypercube] warning: random projections expect L2 metric\n";
    }
}

SearchResult HypercubeSearch::search(const Vector& query,
                                     const Params& params,
                                     int query_id) const {
    auto t0 = Clock::now();
    SearchResult res;
    res.query_id = query_id;

    if (dataset_.empty() || space_dim_ == 0 || query.values.size() != space_dim_) {
        res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return res;
    }

    const int neighbours_requested = params.N > 0 ? params.N : 1;
    const size_t candidate_limit =
        max_candidates_ > 0 ? static_cast<size_t>(max_candidates_) : std::numeric_limits<size_t>::max();

    const uint32_t start_bucket = hash_vector(query.values);

    std::queue<uint32_t> agenda;
    std::unordered_set<uint32_t> visited_buckets;
    agenda.push(start_bucket);
    visited_buckets.insert(start_bucket);

    std::unordered_set<int> seen_points;
    size_t examined = 0;
    int probes_examined = 0;
    bool stop = false;

    using Candidate = std::pair<double, int>;
    std::priority_queue<Candidate> best;
    std::vector<std::pair<int, double>> range_hits;

    while (!agenda.empty() && probes_examined < max_probes_ && !stop) {
        uint32_t current = agenda.front();
        agenda.pop();
        ++probes_examined;

        auto it = cube_.find(current);
        if (it != cube_.end()) {
            for (int idx : it->second) {
                if (!seen_points.insert(idx).second) {
                    continue;
                }

                double dist = metrics::distance(dataset_[static_cast<size_t>(idx)].values,
                                                query.values,
                                                metrics::GLOBAL_METRIC_CFG);
                ++examined;

                if (neighbours_requested > 0) {
                    best.emplace(dist, idx);
                    if (static_cast<int>(best.size()) > neighbours_requested) {
                        best.pop();
                    }
                }

                if (params.enable_range && params.R > 0.0 && dist <= params.R) {
                    range_hits.emplace_back(idx, dist);
                }

                if (examined >= candidate_limit) {
                    stop = true;
                    break;
                }
            }
        }

        if (probes_examined < max_probes_) {
            for (int bit = 0; bit < kproj_; ++bit) {
                uint32_t neighbour = current ^ (1u << bit);
                if (visited_buckets.insert(neighbour).second) {
                    agenda.push(neighbour);
                }
            }
        }
    }

    std::vector<Candidate> ordered;
    ordered.reserve(best.size());
    while (!best.empty()) {
        ordered.push_back(best.top());
        best.pop();
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const Candidate& a, const Candidate& b) { return a.first < b.first; });

    for (const auto& c : ordered) {
        res.neighbor_ids.push_back(c.second);
        res.distances.push_back(static_cast<float>(c.first));
    }

    std::sort(range_hits.begin(), range_hits.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    for (const auto& hit : range_hits) {
        res.range_neighbor_ids.push_back(hit.first);
        res.range_distances.push_back(static_cast<float>(hit.second));
    }

    auto t1 = Clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

uint32_t HypercubeSearch::hash_vector(const std::vector<double>& vec) const {
    uint32_t code = 0;
    for (int i = 0; i < kproj_; ++i) {
        double dot = 0.0;
        for (uint32_t d = 0; d < space_dim_; ++d) {
            dot += vec[d] * projections_[static_cast<size_t>(i)][d];
        }
        double value = (dot + offsets_[static_cast<size_t>(i)]) / w_;
        int cell = static_cast<int>(std::floor(value));
        if (coin_flip(i, cell)) {
            code |= (1u << i);
        }
    }
    return code;
}

bool HypercubeSearch::coin_flip(int function_index, int cell) const {
    const uint64_t key =
        (static_cast<uint64_t>(function_index) << 32) ^
        static_cast<uint64_t>(static_cast<int64_t>(cell));
    const uint64_t rnd = splitmix64(key ^ static_cast<uint64_t>(seed_));
    return (rnd & 1ull) != 0ull;
}

uint64_t HypercubeSearch::splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}