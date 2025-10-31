#ifndef TRUTH_CACHE_H
#define TRUTH_CACHE_H

#include <cstdint>
#include <filesystem>
#include <vector>

#include "../algorithms/search_algorithm.h"
#include "../utils/args_parser.h"

namespace truth_cache {
uint64_t hash_vector_list(const std::vector<Vector>& data);
std::filesystem::path cache_path_for(const Args& args,
                                     uint64_t dataset_hash,
                                     uint64_t query_hash);
bool load(const std::filesystem::path& path,
          std::vector<SearchResult>& results,
          double& total_time_ms);
void save(const std::filesystem::path& path,
          const std::vector<SearchResult>& results,
          double total_time_ms);
} // namespace truth_cache
#endif // TRUTH_CACHE_H