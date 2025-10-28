#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <cmath>
#include <string>

namespace metrics {

enum class MetricType {
    L1,  // Manhattan
    L2   // Euclidean
};

struct MetricConfig {
    MetricType type = MetricType::L2;
};

double manhattan(const std::vector<float>& a, const std::vector<float>& b);
double euclidean(const std::vector<float>& a, const std::vector<float>& b);
double distance(const std::vector<float>& a, const std::vector<float>& b, const MetricConfig& cfg);
MetricConfig parse_metric_type(const std::string& name);

} // namespace metrics

#endif // METRICS_H