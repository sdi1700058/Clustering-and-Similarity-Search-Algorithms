#ifndef IVFFLAT_SEARCH_H
#define IVFFLAT_SEARCH_H

#include "search_algorithm.h"
#include <unordered_map>
#include <vector>
#include <random>
#include <cstdint>

struct IVFFlatParams {
    int seed = 1;
    int kclusters = 50;
    int nprobe = 5;
    int N = 1;
    double R = 2000.0;
    
};

class IVFFlatSearch : public SearchAlgorithm {
private:

    IVFFlatParams p;
    std::mt19937 rng;
    
    std::vector<Vector> data;
    std::vector<Vector> subset_data;
    
    std::vector<Vector> centroids; 
    std::vector<int> assigned_centroid;
    
    std::vector<std::vector<std::pair<int, Vector>>> IL;

    int space_dim = 0;
    int n_points = 0;
    bool index_built = false;

    // Helper Functions
    int nearest_centroid(const Vector& vec);
    int second_nearest_centroid(const Vector& vec);

    // Clustering
    void random_subset();
    void initialization(); // K-Means++ initialization
    void update();         // Centroid update (K-Medians)
    int assignment_lloyds(); // Assigns subset vectors to nearest cluster


public:
    IVFFlatSearch() : rng(p.seed) {} 

    void build_index(const std::vector<Vector>& dataset) override;
    void configure(const Args& args) override;

    // Search
    SearchResult search(const Vector& query, const Params& params, int query_id) const;

    // Utility/Getter methods
    std::vector<Vector> get_centroids();
    std::vector<std::vector<int>> get_centroids_map();

    // Evaluation
    std::pair<std::vector<double>, double> compute_silhouette_fast();
    std::pair<std::vector<double>, double> compute_silhouette();

    std::string name() const override { return "IVFFlat"; }
};
#endif // IVFFLAT_SEARCH_H