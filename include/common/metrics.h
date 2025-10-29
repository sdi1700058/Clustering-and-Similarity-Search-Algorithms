#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>

namespace metrics {
    enum class MetricType { L1, L2 };
    struct MetricConfig { MetricType type = MetricType::L2; };

    extern MetricConfig GLOBAL_METRIC_CFG;

    double manhattan(const std::vector<double>& a, const std::vector<double>& b);
    double euclidean(const std::vector<double>& a, const std::vector<double>& b);
    double distance(const std::vector<double>& a, const std::vector<double>& b, const MetricConfig& cfg);
    MetricConfig parse_metric_type(const std::string& name);
    void set_global_config(const MetricConfig& cfg);
}

#endif // METRICS_H