#include <iostream>
#include <unordered_set>
#include <algorithm>

#include "../../include/common/evaluation_metrics.h"

EvalResults evaluate_results(
    const std::vector<SearchResult>& approx,
    const std::vector<SearchResult>& truth,
    int N,
    double total_time_ms
) {
    EvalResults r;
    int qcount = (int)approx.size();
    if (qcount == 0) return r;

    double af_sum = 0.0, recall_sum = 0.0, t_sum = 0.0;

    for (size_t i = 0; i < approx.size(); ++i) {
        const auto& a = approx[i];
        const auto& t = truth[i];
        if (t.distances.empty() || a.distances.empty()) continue;

        double AF = a.distances[0] / t.distances[0];
        af_sum += AF;

        // Recall@N
        std::unordered_set<int> true_ids(t.neighbor_ids.begin(), t.neighbor_ids.begin() + std::min(N, (int)t.neighbor_ids.size()));
        int hits = 0;
        for (int id : a.neighbor_ids) if (true_ids.count(id)) hits++;
        recall_sum += (double)hits / N;

        t_sum += a.time_ms;
    }

    r.average_AF = af_sum / qcount;
    r.recall_at_N = recall_sum / qcount;
    r.qps = 1000.0 * qcount / total_time_ms;
    r.tApproxAvg = t_sum / qcount;
    r.tTrueAvg = 0.0; // placeholder

    std::cout << "[Eval] AF=" << r.average_AF
              << " Recall@N=" << r.recall_at_N
              << " QPS=" << r.qps
              << " tAvg=" << r.tApproxAvg << "ms\n";

    return r;
}