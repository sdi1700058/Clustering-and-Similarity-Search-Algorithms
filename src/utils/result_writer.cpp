#include <fstream>
#include <iomanip>
#include <iostream>

#include "../../include/utils/result_writer.h"

void write_results(const std::vector<SearchResult>& results, const std::string& path, const std::string& method_name) {
    std::ofstream out(path);
    if (!out.is_open()) { std::cerr << "[Writer] ERROR: cannot open " << path << std::endl; return; }

    out << method_name << "\n";
    out << std::fixed << std::setprecision(6);

    double total_time_approx = 0.0;
    for (const auto &r : results) {
        out << "Query: " << r.query_id << "\n";
        for (size_t i = 0; i < r.neighbor_ids.size(); ++i) {
            out << "Nearest neighbor-" << (i+1) << ": " << r.neighbor_ids[i] << "\n";
            out << "distanceApproximate: " << r.distances[i] << "\n";
            out << "distanceTrue: " << r.distances[i] << "\n"; // Since brute-force == true in this checkpoint
        }
        out << "R-near neighbors:\n"; // empty in brute checkpoint (range not computed here)
        out << "Average AF: 1.0\n";   // AF = approxDist / trueDist => 1.0 for brute
        out << "Recall@N: 1.0\n";     // perfect recall for brute
        out << "QPS: 0.0\n";          // filled later by benchmarking stage
        out << "tApproximateAverage: " << r.time_ms << "\n";
        out << "tTrueAverage: " << r.time_ms << "\n";
        out << "----------------------------------------\n";
        total_time_approx += r.time_ms;
    }
    std::cout << "[Writer] Results written to " << path << " (method: " << method_name << ").\n";
}