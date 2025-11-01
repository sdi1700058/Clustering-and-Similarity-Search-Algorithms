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
    
    std::cout << "[LSH] Built " << p.L << " hash tables for " << dataset.size() << " vectors (space_dim=" << space_dim << ")\n";
}

void LSHSearch::build_hashes() {
    std::normal_distribution<double> gaussian(0.0, 1.0);
    std::uniform_real_distribution<double> distribution(0.0, p.w);

	for (int i = 0; i < p.L; i++) {
        std::vector<std::pair<std::vector<double>, double>> hlist;
        for (int j = 0; j < p.k; j++) {
            std::vector<double> a(space_dim);
            for (int d = 0; d < space_dim; d++)
                a[d] = gaussian(rng);

            double b = distribution(rng);

            hlist.push_back({a,b});
        }

		amplified_hash_fns.push_back(hlist);
	}
}

void LSHSearch::initialize() {
    lsh_tables.clear();
    lsh_tables.reserve(p.L);

	// initialize all of our hash tables
	for (int i = 0; i < p.L; i++) {
        std::unordered_map<int, std::list<int>> table; 
        lsh_tables.push_back(std::move(table));
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
};

int LSHSearch::assign_to_bucket(const std::vector<std::pair<std::vector<double>, double>>& amplified_fn, const Vector& x) const {

    int64_t combined_hash = 0;

    for (int i = 0; i < p.k; i++) {
        const auto& a = amplified_fn[i].first; // random vector
        double b = amplified_fn[i].second;     // offset

        double dot_product = 0.0;
        for (int d = 0; d < space_dim; d++)
            dot_product += a[d] * x.values[d];

        int h = static_cast<int>(std::floor((dot_product + b) / p.w));

        // Combine hashes
        combined_hash = combined_hash * 31 + h;  // simple hash combiner
    }
// Use std::hash to map to a non-negative bucket
    size_t bucket_id = std::hash<int64_t>{}(combined_hash) % (n_points / 4); // table_size = number of buckets
    return static_cast<int>(bucket_id);
    // return modulo(static_cast<int>(combined_hash), p.M);
}

SearchResult LSHSearch::search(const Vector& query, const Params& params, int query_id) const {
    auto t0 = std::chrono::high_resolution_clock::now();
    SearchResult res;
    res.query_id = query_id;  // optional placeholder if needed

    if (amplified_hash_fns.empty() || data.empty() || space_dim == 0) {
        res.time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        return res;
    }

    std::vector<std::pair<int, double>> b; // (index, distance)
    b.reserve(n_points);

    // 1. Traverse LSH tables
    for (size_t table_idx = 0; table_idx < amplified_hash_fns.size(); ++table_idx) {
        const auto& amplified_fn = amplified_hash_fns[table_idx];

        int bucket_id = assign_to_bucket(amplified_fn, query);
        // const auto& bucket = lsh_tables.at(table_idx)[bucket_id];

        // 2. Collect candidate distances
        // for (int i : bucket) {
        //     double dist = metrics::distance(
        //         query.values, 
        //         data[i].values, 
        //         metrics::GLOBAL_METRIC_CFG
        //     );

        //     b.push_back({i, dist});
        // }
        auto bucket = lsh_tables[table_idx].find(bucket_id);
        if (bucket != lsh_tables[table_idx].end()) {
            
            for (int i : bucket->second) {
                double dist = metrics::distance(
                    query.values, 
                    data[i].values, 
                    metrics::GLOBAL_METRIC_CFG
                );

                b.push_back({i, dist});
            }
        }
    }

    // 3. Find the R nearest b
    if (!b.empty()) {
        
        std::partial_sort(b.begin(), b.begin() + params.N, b.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
        
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
