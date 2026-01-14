#include <cassert>
#include <iostream>
#include <cmath>

#include "../../include/common/metrics.h"

namespace metrics {
    MetricConfig GLOBAL_METRIC_CFG;

    double manhattan(const std::vector<double>& a, const std::vector<double>& b) {
        assert(a.size() == b.size());
        double s = 0.0;
        for (size_t i=0;i<a.size();++i) s += std::fabs(a[i]-b[i]);
        return s;
    }

    double euclidean(const std::vector<double>& a, const std::vector<double>& b) {
        assert(a.size() == b.size());
        double s = 0.0;
        for (size_t i=0;i<a.size();++i) {
            double d = a[i]-b[i];
            s += d*d;
        }
        return std::sqrt(s);
    }

    double cosine_dist(const std::vector<double>& a, const std::vector<double>& b) {
        assert(a.size() == b.size());
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a == 0.0 || norm_b == 0.0) return 1.0; // Max distance if zero vector
        
        // Cosine Similarity = dot / (||a|| * ||b||)
        // Cosine Distance = 1 - Similarity
        return 1.0 - (dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    double distance(const std::vector<double>& a, const std::vector<double>& b, const MetricConfig& cfg) {
        switch (cfg.type) {
            case MetricType::L1: return manhattan(a,b);
            case MetricType::L2: return euclidean(a,b);
            case MetricType::COSINE: return cosine_dist(a,b);
            default: return euclidean(a,b);
        }
    }

    MetricConfig parse_metric_type(const std::string& name) {
        MetricConfig cfg;
        if (name == "l1" || name == "L1" || name == "manhattan") cfg.type = MetricType::L1;
        else if (name == "cosine" || name == "cos") cfg.type = MetricType::COSINE;
        else cfg.type = MetricType::L2;
        return cfg;
    }

    void set_global_config(const MetricConfig& cfg) { GLOBAL_METRIC_CFG = cfg; }

} // namespace metrics