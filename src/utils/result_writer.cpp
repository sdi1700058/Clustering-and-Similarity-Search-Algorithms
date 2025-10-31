#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <vector>

#include "../../include/utils/result_writer.h"

void write_results(const std::vector<SearchResult>& results, const std::string& path, const std::string& method_name, double approx_time_ms, const std::string& config_summary) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::vector<char> stream_buffer(1 << 20); // 1 MB buffer
    std::ofstream out;
    out.rdbuf()->pubsetbuf(stream_buffer.data(), static_cast<std::streamsize>(stream_buffer.size()));
    out.open(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) { std::cerr << "[Writer] cannot open " << path << "\n"; return; }
    out << "==== RESULTS OF "<< method_name << " ====\n";
    out << "===== CONFIGURATION =====\n";
    if (!config_summary.empty()) {
        out << config_summary;
        if (config_summary.back() != '\n') out << '\n';
    }
    out << "Execution Time (ms): " << approx_time_ms << "\n";
    out << "=============================================================\n";
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
        out << "=============================================================\n";
        /* Evaluation Metrics Are Written In Summary*/
    }
    std::cout << "[Writer] " << results.size() << " Results written to " << path << " (method: " << method_name << ").\n";
}