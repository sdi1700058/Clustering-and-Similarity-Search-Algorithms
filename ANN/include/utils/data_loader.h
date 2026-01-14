#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <iostream>

#include "../algorithms/search_algorithm.h"

namespace data_loader {

std::vector<Vector> load_dataset(const std::string& path, const std::string& type);
std::vector<Vector> load_queries(const std::string& path, const std::string& type);

}

#endif // DATA_LOADER_H
