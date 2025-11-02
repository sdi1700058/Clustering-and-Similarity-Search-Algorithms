#ifndef IVFPQ_SEARCH_H
#define IVFPQ_SEARCH_H

#include "search_algorithm.h"
#include <cstdint>
#include <random>
#include <vector>

struct IVFPQParams {
    int seed = 1;
    int kclusters = 50;
    int nprobe = 5;
    int M = 16;
    int nbits = 8;
    int N = 1;
    double R = 2000.0;
};

class IVFPQSearch : public SearchAlgorithm {
private:
    IVFPQParams p;
    std::mt19937 rng;

    std::vector<Vector> data;
    std::vector<Vector> subset_data;
    std::vector<Vector> centroids;
    std::vector<int> subset_assignments_;
    std::vector<int> data_assignments_;

    std::vector<std::vector<int>> inverted_lists_;
    std::vector<std::vector<std::uint8_t>> point_codes_;
    std::vector<std::vector<Vector>> pq_codebooks_;

    int space_dim_ = 0;
    int subvector_dim_ = 0;
    int n_points_ = 0;
    int codebook_size_ = 0;
    bool index_built = false;

    // Helper functions for coarse clustering
    void random_subset();
    void initialization();
    int assignment_lloyds();
    void update();
    int nearest_centroid(const Vector& vec) const;
    int second_nearest_centroid(const Vector& vec) const;

    // Product Quantization helpers
    void build_pq_codebooks();
    std::vector<double> compute_residual(const Vector& vec, int centroid_idx) const;
    std::vector<std::uint8_t> encode_point(const Vector& vec, int centroid_idx) const;

public:
    IVFPQSearch() : rng(p.seed) {}

    void configure(const Args& args) override;
    void build_index(const std::vector<Vector>& dataset) override;

    SearchResult search(const Vector& query, const Params& params, int query_id) const override;

    std::vector<Vector> get_centroids() const { return centroids; }
    std::vector<std::vector<int>> get_centroids_map() const;

    std::pair<std::vector<double>, double> compute_silhouette_fast() const;
    std::pair<std::vector<double>, double> compute_silhouette() const;

    std::string name() const override { return "IVFPQ"; }
};

#endif // IVFPQ_SEARCH_H