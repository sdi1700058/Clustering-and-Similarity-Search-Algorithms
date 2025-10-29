#ifndef PARALLEL_RUNNER_H
#define PARALLEL_RUNNER_H

#include <vector>

#include "../algorithms/search_algorithm.h"

std::vector<SearchResult> run_parallel_search(
    const SearchAlgorithm* algo,
    const std::vector<Vector>& queries,
    int num_threads,
    const Params& params
);

#endif // PARALLEL_RUNNER_H