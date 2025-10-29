#ifndef HYPERCUBE_SEARCH_H
#define HYPERCUBE_SEARCH_H

#include "search_algorithm.h"
#include <unordered_map>
#include <vector>
#include <random>
#include <cstdint>

// hypercube parameters
class HypercubeSearch : public SearchAlgorithm {
public:
    void configure(const Args& args) override;
    void build_index(const std::vector<Vector>& dataset) override;
    SearchResult search(const Vector& query, const Params& params, int query_id) const override;
    std::string name() const override { return "Hypercube"; }

private:
    using Bucket = std::vector<int>;

    int seed_ = 1;
    int kproj_ = 14;
    int max_candidates_ = 10;
    int max_probes_ = 2;
    double w_ = 4.0;

    uint32_t space_dim_ = 0;
    std::vector<Vector> dataset_;
    std::unordered_map<uint32_t, Bucket> cube_;
    std::vector<std::vector<double>> projections_;
    std::vector<double> offsets_;

    uint32_t hash_vector(const std::vector<double>& vec) const;
    bool coin_flip(int function_index, int cell) const;
    static uint64_t splitmix64(uint64_t x);
};
#endif // HYPERCUBE_SEARCH_H