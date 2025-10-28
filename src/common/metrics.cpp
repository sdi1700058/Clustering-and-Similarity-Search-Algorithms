#include <cassert>
#include <iostream>

#include "../../include/common/metrics.h"

namespace metrics {

double manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        s += std::fabs(a[i] - b[i]);
    return s;
}

double euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

double distance(const std::vector<float>& a, const std::vector<float>& b, const MetricConfig& cfg) {
    switch (cfg.type) {
        case MetricType::L1: return manhattan(a, b);
        case MetricType::L2: return euclidean(a, b);
    }
    return euclidean(a, b);
}

MetricConfig parse_metric_type(const std::string& name) {
    MetricConfig cfg;
    if (name == "l1" || name == "manhattan") cfg.type = MetricType::L1;
    else cfg.type = MetricType::L2;
    return cfg;
}

} // namespace metrics
