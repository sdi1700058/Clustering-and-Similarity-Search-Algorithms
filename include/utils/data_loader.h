#ifndef DATA_LOADER_H
#define DATA_LOADER_H


#include <string>
#include <vector>

#include "../algorithms/search_algorithm.h"

/*
 NOTE: CURRENT CHECKPOINT: simple ASCII loader is implemented (format below).
 TODO: Add binary MNIST (big-endian) and SIFT .fvecs (little-endian) loaders
 to conform strictly to the assignment.
*/

std::vector<Vector> load_vectors(const std::string& path);

#endif // DATA_LOADER_H
