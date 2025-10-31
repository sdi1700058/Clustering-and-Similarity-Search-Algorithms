#include <fstream>
#include <sstream>
#include <cstring>
#include <functional>
#include <iostream>

#include "../../include/utils/truth_cahce.h"
#include "../../include/common/metrics.h"

namespace {
constexpr uint32_t TRUTH_MAGIC = 0x54525554; // 'TRUT'
constexpr uint32_t TRUTH_VERSION = 1;

uint64_t hash_combine64(uint64_t seed, uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

uint64_t double_bits(double value) {
    uint64_t bits{};
    std::memcpy(&bits, &value, sizeof(double));
    return bits;
}
} // namespace

namespace truth_cache {

uint64_t hash_vector_list(const std::vector<Vector>& data) {
    uint64_t seed = 0xcbf29ce484222325ULL;
    std::hash<double> h;
    for (const auto& vec : data) {
        seed = hash_combine64(seed, static_cast<uint64_t>(vec.values.size()));
        for (double v : vec.values) {
            seed = hash_combine64(seed, static_cast<uint64_t>(h(v)));
        }
    }
    return seed;
}

std::filesystem::path cache_path_for(const Args& args,
                                     uint64_t dataset_hash,
                                     uint64_t query_hash) {
    uint64_t key = 0xcbf29ce484222325ULL;
    key = hash_combine64(key, dataset_hash);
    key = hash_combine64(key, query_hash);
    key = hash_combine64(key, std::hash<std::string>{}(args.dataset_path));
    key = hash_combine64(key, std::hash<std::string>{}(args.query_path));
    key = hash_combine64(
        key,
        static_cast<uint64_t>(static_cast<int>(metrics::GLOBAL_METRIC_CFG.type)));
    key = hash_combine64(key, static_cast<uint64_t>(args.N));
    key = hash_combine64(key, double_bits(args.R));
    key = hash_combine64(key, static_cast<uint64_t>(args.range));
    key = hash_combine64(key, TRUTH_VERSION);

    std::ostringstream oss;
    oss << std::hex << key;

    std::filesystem::path out_path(args.output_path);
    std::filesystem::path base_dir = out_path.has_parent_path()
        ? out_path.parent_path()
        : std::filesystem::path("output");
    return base_dir / "cache" / ("truth_" + oss.str() + ".bin");
}

bool load(const std::filesystem::path& path,
          std::vector<SearchResult>& results,
          double& total_time_ms) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    uint32_t magic = 0, version = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!in || magic != TRUTH_MAGIC || version != TRUTH_VERSION) {
        std::cerr << "[Cache] Invalid truth cache header in " << path << "\n";
        return false;
    }

    uint64_t count = 0;
    in.read(reinterpret_cast<char*>(&total_time_ms), sizeof(total_time_ms));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in) {
        std::cerr << "[Cache] Failed to read truth cache metadata from " << path << "\n";
        return false;
    }

    results.clear();
    results.resize(static_cast<size_t>(count));

    for (uint64_t i = 0; i < count; ++i) {
        int32_t query_id = -1;
        uint32_t neighbor_count = 0;
        uint32_t range_count = 0;
        double time_ms = 0.0;

        in.read(reinterpret_cast<char*>(&query_id), sizeof(query_id));
        in.read(reinterpret_cast<char*>(&neighbor_count), sizeof(neighbor_count));
        in.read(reinterpret_cast<char*>(&range_count), sizeof(range_count));
        in.read(reinterpret_cast<char*>(&time_ms), sizeof(time_ms));

        if (!in) {
            std::cerr << "[Cache] Corrupted truth cache entry in " << path << "\n";
            return false;
        }

        SearchResult res;
        res.query_id = query_id;
        res.time_ms = time_ms;

        res.neighbor_ids.resize(neighbor_count);
        res.distances.resize(neighbor_count);
        in.read(reinterpret_cast<char*>(res.neighbor_ids.data()),
                static_cast<std::streamsize>(neighbor_count * sizeof(int)));
        in.read(reinterpret_cast<char*>(res.distances.data()),
                static_cast<std::streamsize>(neighbor_count * sizeof(float)));

        res.range_neighbor_ids.resize(range_count);
        res.range_distances.resize(range_count);
        in.read(reinterpret_cast<char*>(res.range_neighbor_ids.data()),
                static_cast<std::streamsize>(range_count * sizeof(int)));
        in.read(reinterpret_cast<char*>(res.range_distances.data()),
                static_cast<std::streamsize>(range_count * sizeof(float)));

        if (!in) {
            std::cerr << "[Cache] Failed while reading truth cache data from " << path << "\n";
            return false;
        }

        results[static_cast<size_t>(i)] = std::move(res);
    }

    std::cout << "[Cache] Loaded ground truth results from " << path << "\n";
    return true;
}

void save(const std::filesystem::path& path,
          const std::vector<SearchResult>& results,
          double total_time_ms) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        std::cerr << "[Cache] Cannot ensure directory for " << path << ": " << ec.message() << "\n";
        return;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "[Cache] Failed to open " << path << " for writing\n";
        return;
    }

    uint32_t magic = TRUTH_MAGIC;
    uint32_t version = TRUTH_VERSION;
    uint64_t count = static_cast<uint64_t>(results.size());

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&total_time_ms), sizeof(total_time_ms));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& res : results) {
        int32_t query_id = res.query_id;
        uint32_t neighbor_count = static_cast<uint32_t>(res.neighbor_ids.size());
        uint32_t range_count = static_cast<uint32_t>(res.range_neighbor_ids.size());
        double time_ms = res.time_ms;

        out.write(reinterpret_cast<const char*>(&query_id), sizeof(query_id));
        out.write(reinterpret_cast<const char*>(&neighbor_count), sizeof(neighbor_count));
        out.write(reinterpret_cast<const char*>(&range_count), sizeof(range_count));
        out.write(reinterpret_cast<const char*>(&time_ms), sizeof(time_ms));

        out.write(reinterpret_cast<const char*>(res.neighbor_ids.data()),
                  static_cast<std::streamsize>(neighbor_count * sizeof(int)));
        out.write(reinterpret_cast<const char*>(res.distances.data()),
                  static_cast<std::streamsize>(neighbor_count * sizeof(float)));
        out.write(reinterpret_cast<const char*>(res.range_neighbor_ids.data()),
                  static_cast<std::streamsize>(range_count * sizeof(int)));
        out.write(reinterpret_cast<const char*>(res.range_distances.data()),
                  static_cast<std::streamsize>(range_count * sizeof(float)));
    }

    std::cout << "[Cache] Stored ground truth results at " << path << "\n";
}

} // namespace truth_cache