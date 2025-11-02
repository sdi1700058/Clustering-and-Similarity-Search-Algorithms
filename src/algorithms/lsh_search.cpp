#include <iostream>
#include <unordered_set>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "../../include/algorithms/lsh_search.h"
#include "../../include/utils/args_parser.h"

void LSHSearch::configure(const Args& args) {
    p.seed = args.seed;
    p.k = args.k;
    p.L = args.L;
    p.w = args.w;
    p.N = args.N;
    p.R = args.R;
    rng.seed(args.seed);
}

void LSHSearch::build_index(const std::vector<Vector>& dataset) {
    data = dataset;
    space_dim = dataset.empty() ? 0 : static_cast<int>(dataset.front().values.size());
    n_points = dataset.empty() ? 0 : static_cast<int>(dataset.size());

    build_hashes();
    initialize();
    build_tables();

    std::cout << "[LSH] Built " << p.L << " hash tables for " 
              << dataset.size() << " vectors (space_dim=" 
              << space_dim << ")\n";
}

void LSHSearch::build_hashes() {
    std::uniform_real_distribution<double> distribution(0.0, p.w);

    for (int i = 0; i < p.L; i++) {
        std::vector<std::vector<std::pair<int, double>>> hlist;
        for (int j = 0; j < p.k; j++) {
            std::vector<std::pair<int, double>> const_values(space_dim);
            // s, const_m_values
            for (int l = 0; l < space_dim; l++) {
                const_values[l].first = distribution(rng);
                const_values[l].second = modular_power(p.m, l, p.M);
            }
            hlist.push_back(const_values);
        }
        amplified_hash_fns.push_back(hlist);
    }
}

void LSHSearch::initialize() {
    lsh_tables.clear();
    lsh_tables.reserve(p.L);

    // initialize all of our hash tables
    for (int i = 0; i < p.L; i++) {
        // for each hash table
        std::vector<std::list<int>> current_hash(p.M);
        lsh_tables.push_back(std::move(current_hash));
    }
}

void LSHSearch::build_tables() {
    for (size_t table_idx = 0; table_idx < amplified_hash_fns.size(); ++table_idx) {
        const auto& amplified_fn = amplified_hash_fns[table_idx];
        for (int i = 0; i < n_points; ++i) {
            int bucket = assign_to_bucket(amplified_fn, data[i]);
            lsh_tables[table_idx][bucket].push_back(i);
        }
    }
}

int LSHSearch::modulo(int a, int b) const {
    int c = a % b;
    return (c < 0) ? c + b : c;
}

// executes the operation (x^y) mod p
int LSHSearch::modular_power(int x, int y, int p) {
    int res = 1;
    x = x % p;

    while (y > 0) {
        if (y & 1)
            res = (res * x) % p;
        y = y >> 1;
        x = (x * x) % p;
    }
    return res;
}

int LSHSearch::assign_to_bucket(
    const std::vector<std::vector<std::pair<int, double>>>& amplified_fn,
    const Vector& x) 
const {
    int result = 0;

    for (auto& hash_it : amplified_fn) {
        uint32_t h = 0;
        std::vector<int> a(space_dim, 0);

        for (int i = 0; i < space_dim; i++) {
            a[i] = floor((x.values[i] - hash_it[i].first) / (p.w * 1.0));
        }

        for (int i = 0; i < space_dim; i++) {
            h += modulo(modulo(a.at(space_dim - i - 1), p.M) * hash_it[i].second, p.M);
        }

        result = (result << 32 / p.k) | h % p.M;
    }

    return modulo(result, p.M);
}

SearchResult LSHSearch::search(const Vector& query, const Params& params, int query_id) const {
    auto t0 = std::chrono::high_resolution_clock::now();
    SearchResult res;
    res.query_id = query_id;

    if (amplified_hash_fns.empty() || data.empty() || space_dim == 0) {
        res.time_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0
        ).count();
        return res;
    }

    std::vector<std::pair<int, double>> b; // (index, distance)
    b.reserve(n_points);
    int table_idx = 0;

    // 1. Traverse LSH tables
    for (const auto& it : amplified_hash_fns) {
        int bucket_id = assign_to_bucket(it, query);
        const auto& bucket = lsh_tables.at(table_idx)[bucket_id];

        // 2. Collect candidate distances
        for (int vec_idx : bucket) {
            double dist = metrics::distance(
                query.values,
                data[vec_idx].values,
                metrics::GLOBAL_METRIC_CFG
            );
            b.push_back({vec_idx, dist});
        }
        ++table_idx;
    }

    // 3. Find the R nearest
    if (!b.empty()) {
        std::partial_sort(
            b.begin(),
            b.begin() + params.N,
            b.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            }
        );

        int topK = std::min(params.N, static_cast<int>(b.size()));
        for (int i = 0; i < topK; ++i) {
            res.neighbor_ids.push_back(b[i].first);
            res.distances.push_back(static_cast<float>(b[i].second));
        }

        if (params.enable_range && params.R > 0.0) {
            for (const auto& cand : b) {
                if (cand.second <= params.R) {
                    res.range_neighbor_ids.push_back(cand.first);
                    res.range_distances.push_back(static_cast<float>(cand.second));
                } else {
                    break;
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}