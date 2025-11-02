#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <vector>

#include "../../include/utils/result_writer.h"
#include "../../include/common/evaluation_metrics.h"

void write_results(const std::vector<SearchResult>& results, const std::string& path, const std::string& method_name, double approx_time_ms, const std::string& config_summary, const std::vector<SearchResult>* truth_results, const EvalResults* eval_summary) {
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
    if (eval_summary) {
        out << "===== EVALUATION =====\n";
        out << "Average AF: " << eval_summary->average_AF << "\n";
        out << "Recall@N: " << eval_summary->recall_at_N << "\n";
        out << "QPS: " << eval_summary->qps << "\n";
        out << "tApproximateAverage: " << eval_summary->tApproxAvg << "\n";
        out << "tTrueAverage: " << eval_summary->tTrueAvg << "\n";
    }
    out << "=============================================================\n";
    out << std::fixed << std::setprecision(6);
    for (size_t query_idx = 0; query_idx < results.size(); ++query_idx) {
        const auto& r = results[query_idx];
        const SearchResult* truth = (truth_results && query_idx < truth_results->size())
                                        ? &(*truth_results)[query_idx]
                                        : nullptr;
        out << "Query: " << r.query_id << "\n";
        int K = static_cast<int>(r.neighbor_ids.size());
        for (int i = 0; i < K; ++i) {
            float approx_dist = (i < static_cast<int>(r.distances.size())) ? r.distances[i] : 0.0f;
            float true_dist = approx_dist;
            if (truth && i < static_cast<int>(truth->distances.size())) {
                true_dist = truth->distances[i];
            }
            out << "Nearest neighbor-" << (i + 1) << ": " << r.neighbor_ids[i] << "\n";
            out << "distanceApproximate: " << approx_dist << "\n";
            out << "distanceTrue: " << true_dist << "\n";
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