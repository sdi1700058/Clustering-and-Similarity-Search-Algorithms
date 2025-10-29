#ifndef IVFFLAT_SEARCH_H
#define IVFFLAT_SEARCH_H

#include "search_algorithm.h"

struct IVFFlatParams {
    int seed = 1;
    int kclusters = 50;
    int nprobe = 5;
    int N = 1;
    double R = 2000.0;
};

class IVFFlatSearch : public SearchAlgorithm {
private:
    std::vector<Vector> data;
    IVFFlatParams p;
    std::vector<Vector> centroids; // placeholder for clusters
    bool index_built = false;
    // THIS IS A PLACEHOLDER
public:
    void build_index(const std::vector<Vector>& dataset) override;
    SearchResult search(const Vector& query, const Params& params, int query_id) const override;
    void configure(const Args& args) override;
    std::string name() const override { return "IVFFlat"; }
};
#endif // IVFFLAT_SEARCH_H