#ifndef EVALUATION_METRICS_H
#define EVALUATION_METRICS_H

#include "../algorithms/search_algorithm.h"
#include <vector>
#include <string>
#include <numeric>

struct EvalResults {
    double average_AF = 0.0;
    double recall_at_N = 0.0;
    double qps = 0.0;
    double tApproxAvg = 0.0;
    double tTrueAvg = 0.0;
};

EvalResults evaluate_results(
    const std::vector<SearchResult>& approx,
    const std::vector<SearchResult>& truth,
    int N,
    double total_time_ms_approx,
    double total_time_ms_truth
); // namespace eval

#endif // EVALUATION_METRICS_H