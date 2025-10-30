#ifndef RESULT_WRITER_H
#define RESULT_WRITER_H

#include <vector>
#include <string>

#include "../algorithms/search_algorithm.h"

void write_results(const std::vector<SearchResult>& results, 
                   const std::string& path, 
                   const std::string& method_name,
                   double approx_time_ms = 0.0,
                   const std::string& config_summary = {});
#endif // RESULT_WRITER_H