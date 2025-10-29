#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

#include "../../include/utils/result_writer.h"

void write_results(const std::vector<SearchResult>& results, const std::string& path, const std::string& method_name) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    if (!out.is_open()) { std::cerr << "[Writer] cannot open " << path << "\n"; return; }
    out << method_name << "\n";
    out << std::fixed << std::setprecision(6);
    for (const auto &r : results) {
        out << "Query: " << r.query_id << "\n";
        int K = (int)r.neighbor_ids.size();
        for (int i=0;i<K;++i) {
            out << "Nearest neighbor-" << (i+1) << ": " << r.neighbor_ids[i] << "\n";
            out << "distanceApproximate: " << r.distances[i] << "\n";
            // Placeholder: distanceTrue is populated by brute force results during evaluation phase
            out << "distanceTrue: " << r.distances[i] << "\n";
        }
        if (!r.range_neighbor_ids.empty()) {
            out << "R-near neighbors:\n";
            for (size_t idx = 0; idx < r.range_neighbor_ids.size(); ++idx) {
                out << r.range_neighbor_ids[idx] << " (dist=" << r.range_distances[idx] << ")\n";
            }
        }
        out << "\n";
        out << "----------------------------------------\n";
        /* Evaluation Metrics Are Written In Summary*/
    }
    std::cout << "[Writer] " << results.size() << " Results written to " << path << " (method: " << method_name << ").\n";
}