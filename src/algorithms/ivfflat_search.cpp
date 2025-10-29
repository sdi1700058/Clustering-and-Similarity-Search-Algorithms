#include <iostream>
#include <thread>
#include <chrono>
#include <random>

#include "../../include/algorithms/ivfflat_search.h"
#include "../../include/utils/args_parser.h"

void IVFFlatSearch::configure(const Args& args) {
    p.seed = args.seed;
    p.kclusters = args.kclusters;
    p.nprobe = args.nprobe;
    p.N = args.N;
    p.R = args.R;
}

void IVFFlatSearch::build_index(const std::vector<Vector>& dataset) {
    data = dataset;
    // PLACEHOLDER: create empty centroids (real K-means to be implemented later)
    centroids.clear();
    index_built = true;
    std::cout << "[IVFFlat - placeholder] index built with " << data.size() << " vectors, k=" << p.kclusters << "\n";
}

SearchResult IVFFlatSearch::search(const Vector& query, const Params& params, int query_id) const {
    (void)query;
    (void)params;
    SearchResult r; r.query_id = query_id;
    for (int i=0;i<p.N && i<(int)data.size(); ++i) {
        r.neighbor_ids.push_back(i);
        r.distances.push_back(0.0);
    }
    r.time_ms = 1.0;
    return r;
}