#include <iostream>

#include "../../include/utils/algorithm_factory.h"
#include "../../include/algorithms/brute_force_search.h"
#include "../../include/algorithms/dummy_search.h"
#include "../../include/algorithms/lsh_search.h"
#include "../../include/algorithms/hypercube_search.h"
#include "../../include/algorithms/ivfflat_search.h"
#include "../../include/algorithms/ivfpq_search.h"
#include "../../include/utils/args_parser.h"

std::unique_ptr<SearchAlgorithm> create_algorithm(const std::string& name) {
    if (name == "brute") return std::make_unique<BruteForceSearch>();
    if (name == "dummy") return std::make_unique<DummySearch>();
    if (name == "lsh") return std::make_unique<LSHSearch>();
    if (name == "hypercube") return std::make_unique<HypercubeSearch>();
    if (name == "ivfflat") return std::make_unique<IVFFlatSearch>();
    if (name == "ivfpq") return std::make_unique<IVFPQSearch>();
    std::cerr << "[Factory] Unknown algorithm '" << name << "'; falling back to brute.\n";
    return std::make_unique<BruteForceSearch>();
}