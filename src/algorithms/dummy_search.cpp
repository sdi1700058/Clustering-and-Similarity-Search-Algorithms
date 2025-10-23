#include <chrono>
#include <thread>

#include "../../include/algorithms/dummy_search.h"

/*
!PLACEHOLDER IMPLEMENTATION OF A DUMMY SEARCH ALGORITHM!
This serves basically as a test to check if the main program
and the overall structure is working fine (builds and runs successfully).
*/

void DummySearch::build_index(const std::vector<Vector>& dataset) {
    data = dataset;
    std::cout << "[DummySearch] Index built with " << dataset.size() << " vectors.\n";
}

SearchResult DummySearch::search(const Vector& query, const Params& params, int query_id) const {
    auto start = std::chrono::high_resolution_clock::now();

    //Just return the query as its own neighbor
    SearchResult res;
    res.query_id = query_id;
    res.neighbor_ids.push_back(query_id);
    res.distances.push_back(0.0f);

    // Simulate small processing time
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    auto end = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return res;
}
