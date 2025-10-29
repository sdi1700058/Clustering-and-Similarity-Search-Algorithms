#ifndef LSH_SEARCH_H
#define LSH_SEARCH_H

#include "search_algorithm.h"
#include "../common/metrics.h"
#include <random>
#include <unordered_map>
#include <vector>
#include <cstdint>

// LSH parameters
struct LSHParams {
    int seed = 1;
    int k = 4;         // number of hi per g (hash functions per table)
    int L = 5;         // number of tables
    double w = 4.0;    // window length
    int N = 1;
    double R = 2000.0; // default for MNIST; override to 2 for SIFT
};

class LSHSearch : public SearchAlgorithm {
private:
    std::vector<Vector> dataset_;
    LSHParams p;
    struct HashFunction {
        std::vector<double> a;
        double b = 0.0;
    };
    struct HashTable {
        std::vector<HashFunction> g; // size k
        std::unordered_map<std::uint64_t, std::vector<int>> buckets;
    };
    mutable std::mt19937 rng{1};
    std::vector<HashTable> tables;
    int space_dim_ = 0;
    double default_w = 4.0;
    std::uint64_t hash_combine(const std::vector<int>& hvals) const;
    std::vector<int> compute_hashes(const HashTable& table, const Vector& v) const;
    std::uint64_t compute_key(const HashTable& table, const Vector& v) const;
    void build_tables();
public:
    LSHSearch() = default;
    void build_index(const std::vector<Vector>& dataset) override;
    SearchResult search(const Vector& query, const Params& params, int query_id) const override;
    void configure(const Args& args) override;
    std::string name() const override { return "LSH"; }
};

#endif // LSH_SEARCH_H