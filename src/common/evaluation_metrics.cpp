#include <iostream>
#include <unordered_set>
#include <algorithm>

#include "../../include/common/evaluation_metrics.h"

EvalResults evaluate_results(const std::vector<SearchResult>& approx,
                             const std::vector<SearchResult>& truth, int N,
                             double total_time_ms_approx,
                             double total_time_ms_truth) {
    EvalResults r;
    size_t qcount = approx.size();
    if (qcount == 0 || truth.size() != approx.size()) return r;

    double af_sum = 0.0;
    double recall_sum = 0.0;
    double tapprox_sum = 0.0;
    double ttrue_sum = 0.0;

    for (size_t i = 0; i < qcount; ++i) {
        const auto &a = approx[i];
        const auto &t = truth[i];
        if (t.distances.empty() || a.distances.empty()) continue;
        double af = a.distances[0] / (t.distances[0] + 1e-12);
        af_sum += af;

        std::unordered_set<int> true_ids;
        int lim = std::min(N, (int)t.neighbor_ids.size());
        for (int j=0;j<lim;++j) true_ids.insert(t.neighbor_ids[j]);

        int hits = 0;
        for (int id : a.neighbor_ids) if (true_ids.count(id)) ++hits;
        recall_sum += (double)hits / (double)N;
        tapprox_sum += a.time_ms;
        ttrue_sum += t.time_ms;
    }

    r.average_AF = af_sum / (double)qcount;
    r.recall_at_N = recall_sum / (double)qcount;
    r.qps = 1000.0 * ((double)qcount) / (total_time_ms_approx > 0 ? total_time_ms_approx : 1.0);
    const double truth_qps = 1000.0 * ((double)qcount) / (total_time_ms_truth > 0 ? total_time_ms_truth : 1.0);
    r.tApproxAvg = tapprox_sum / (double)qcount;
    r.tTrueAvg = ttrue_sum / (double)qcount;

    std::cout << "[Eval] Average AF=" << r.average_AF
              << " Recall@" << N << "=" << r.recall_at_N
              << " QPS=" << r.qps
              << " TruthQPS=" << truth_qps
              << " tApproxAvg=" << r.tApproxAvg << "ms"
              << " tTrueAvg=" << r.tTrueAvg << "ms\n";
    return r;
}