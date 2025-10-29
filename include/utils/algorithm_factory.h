#ifndef ALGORITHM_FACTORY_H
#define ALGORITHM_FACTORY_H

#include <memory>
#include <string>

#include "../algorithms/search_algorithm.h"

std::unique_ptr<SearchAlgorithm> create_algorithm(const std::string& name);
#endif // ALGORITHM_FACTORY_H