#include <iostream>
#include <thread>
#include <chrono>
#include <random>

#include "../../include/algorithms/ivfpq_search.h"
#include "../../include/utils/args_parser.h"

void IVFPQSearch::configure(const Args& args) {
    p.seed = args.seed;
    p.kclusters = args.kclusters;
    p.nprobe = args.nprobe;
    p.M = args.pq_M;
    p.nbits = args.pq_nbits;
    p.N = args.N;
    p.R = args.R;
}

void IVFPQSearch::build_index(const std::vector<Vector>& dataset) {
    data = dataset;
    centroids.clear();
    index_built = true;
    std::cout << "[IVFPQ - placeholder] index built with " << data.size() << " vectors, k=" << p.kclusters << " M=" << p.M << "\n";
}

SearchResult IVFPQSearch::search(const Vector& query, const Params& params, int query_id) const {
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