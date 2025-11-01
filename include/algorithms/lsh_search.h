#ifndef LSH_SEARCH_H
#define LSH_SEARCH_H

#include "search_algorithm.h"
#include "../common/metrics.h"
#include <random>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <list>

// LSH parameters
struct LSHParams {
    int seed = 1;
    int k = 4;         // number of hi per g (hash functions per table)
    int L = 5;         // number of tables
    double w = 123456;    // window length
    int N = 1;
    double R = 2000.0; // default for MNIST; override to 2 for SIFT

	uint32_t c = 1; 
	uint32_t M = 256;
	uint64_t m = pow(2,32) - 5;
};

class LSHSearch : public SearchAlgorithm {
private:
    LSHParams p;
    std::mt19937 rng;

    std::vector<Vector> data;

    std::vector<std::vector<std::pair<std::vector<double>, double>>> amplified_hash_fns;
    std::vector<std::unordered_map<int, std::list<int>>> lsh_tables;

    int space_dim = 0;
    int n_points = 0;
    bool index_built = false;

    // Hastables
    void build_hashes();
    void initialize();
    void build_tables();
    int modulo(int a, int b) const;
    int modular_power(int x, int y, int p);
    int assign_to_bucket(const std::vector<std::pair<std::vector<double>, double>>& amplified_fn, const Vector& x) const;

public:
    LSHSearch() : rng(p.seed) {} 

    void build_index(const std::vector<Vector>& dataset) override;
    void configure(const Args& args) override;

    // Search
    SearchResult search(const Vector& query, const Params& params, int query_id) const;


    std::string name() const override { return "LSH"; }


};

#endif // LSH_SEARCH_H