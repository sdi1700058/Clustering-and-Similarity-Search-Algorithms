#ifndef IVFPQ_SEARCH_H
#define IVFPQ_SEARCH_H

#include "search_algorithm.h"

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
    std::vector<Vector> data;
    IVFPQParams p;
    std::vector<Vector> centroids; // placeholder for clusters
    bool index_built = false;
    // THIS IS A PLACEHOLDER
public:
    void build_index(const std::vector<Vector>& dataset) override;
    SearchResult search(const Vector& query, const Params& params, int query_id) const override;
    void configure(const Args& args) override;
    std::string name() const override { return "IVFPQ"; }
};
#endif // IVFPQ_SEARCH_H