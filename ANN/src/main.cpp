#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <chrono>

#include "../include/utils/args_parser.h"
#include "../include/algorithms/brute_force_search.h"
#include "../include/algorithms/dummy_search.h"
#include "../include/algorithms/search_algorithm.h"
#include "../include/utils/algorithm_factory.h"
#include "../include/utils/parallel_runner.h"
#include "../include/utils/data_loader.h"
#include "../include/utils/result_writer.h"
#include "../include/common/metrics.h"
#include "../include/common/evaluation_metrics.h"

// --- Helper Functions for Caching Ground Truth ---

void save_ground_truth(const std::string& filename, const std::vector<SearchResult>& results, double total_time_ms) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "[Cache] Error: Could not save to " << filename << "\n";
        return;
    }

    // 1. Save metadata
    out.write(reinterpret_cast<const char*>(&total_time_ms), sizeof(total_time_ms));

    size_t size = results.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // 2. Save results
    for (const auto& res : results) {
        out.write(reinterpret_cast<const char*>(&res.query_id), sizeof(res.query_id));
        out.write(reinterpret_cast<const char*>(&res.time_ms), sizeof(res.time_ms));

        // Neighbors
        size_t n_size = res.neighbor_ids.size();
        out.write(reinterpret_cast<const char*>(&n_size), sizeof(n_size));
        if (n_size > 0) out.write(reinterpret_cast<const char*>(res.neighbor_ids.data()), n_size * sizeof(int));

        // Distances
        size_t d_size = res.distances.size();
        out.write(reinterpret_cast<const char*>(&d_size), sizeof(d_size));
        if (d_size > 0) out.write(reinterpret_cast<const char*>(res.distances.data()), d_size * sizeof(float));

        // Range Neighbors
        size_t rn_size = res.range_neighbor_ids.size();
        out.write(reinterpret_cast<const char*>(&rn_size), sizeof(rn_size));
        if (rn_size > 0) out.write(reinterpret_cast<const char*>(res.range_neighbor_ids.data()), rn_size * sizeof(int));

        // Range Distances
        size_t rd_size = res.range_distances.size();
        out.write(reinterpret_cast<const char*>(&rd_size), sizeof(rd_size));
        if (rd_size > 0) out.write(reinterpret_cast<const char*>(res.range_distances.data()), rd_size * sizeof(float));
    }
    std::cout << "[Cache] Saved ground truth to " << filename << "\n";
}

bool load_ground_truth(const std::string& filename, std::vector<SearchResult>& results, double& total_time_ms) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    // 1. Load total execution time
    in.read(reinterpret_cast<char*>(&total_time_ms), sizeof(total_time_ms));

    // 2. Load number of queries
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    results.resize(size);

    // 3. Load each result
    for (auto& res : results) {
        in.read(reinterpret_cast<char*>(&res.query_id), sizeof(res.query_id));
        in.read(reinterpret_cast<char*>(&res.time_ms), sizeof(res.time_ms));

        size_t count;
        
        // Neighbors
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        res.neighbor_ids.resize(count);
        if (count > 0) in.read(reinterpret_cast<char*>(res.neighbor_ids.data()), count * sizeof(int));

        // Distances
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        res.distances.resize(count);
        if (count > 0) in.read(reinterpret_cast<char*>(res.distances.data()), count * sizeof(float));

        // Range Neighbors
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        res.range_neighbor_ids.resize(count);
        if (count > 0) in.read(reinterpret_cast<char*>(res.range_neighbor_ids.data()), count * sizeof(int));

        // Range Distances
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        res.range_distances.resize(count);
        if (count > 0) in.read(reinterpret_cast<char*>(res.range_distances.data()), count * sizeof(float));
    }
    return true;
}

/* 
    Clustering and Similarity Search Algorithms 

    This is the main program file that does the following:
        - Parses command-line arguments
        - Initializes data structures
        - Splits the data for parallel processing
        - Calls the algorithm that was specified in the arguments
        - Outputs the results to the specified location
        - Cleans up resources

    This program aims to be generic, meaning that the implemetation of the
    algorithms should not affect the main program flow.
*/

int main(int argc, char** argv) {
    std::cout << "=== ANN Framework ===\n";
    Args args = parse_args(argc, argv);

    // Set metric globally
    auto mcfg = metrics::parse_metric_type(args.metric);
    metrics::set_global_config(mcfg);

    // Load dataset and queries
    std::vector<Vector> dataset;
    std::vector<Vector> queries;
    try {
        dataset = data_loader::load_dataset(args.dataset_path, args.type);
        queries = data_loader::load_queries(args.query_path, args.type);
    } catch (const std::exception& e) {
        std::cerr << "[Main] Error loading data: " << e.what() << "\n";
        return 1;
    }

    Params params; params.N = args.N; params.R = args.R; params.enable_range = args.range;
    
    // Create approx algorithm and configure
    auto approx = create_algorithm(args.algo);

    approx->configure(args);
    approx->build_index(dataset);

    // --- Ground Truth Logic with Caching ---
    
    std::vector<SearchResult> truth_results;
    double truth_time_ms = 0.0;

    // 1. Construct filename: val/<type>/truth_<type>_N_<N>_R_<R>.bin
    std::string val_dir = "val/" + args.type;
    std::filesystem::create_directories(val_dir); // Ensure directory exists
    double r_val = args.range ? args.R : 0.0;
    std::string cache_filename = val_dir + "/truth_" + args.type + "_N_" + std::to_string(args.N) + "_R_" + std::to_string(r_val) + "_" + args.metric + ".bin";
    // 2. Try to load from file
    bool loaded = false;
    if (std::filesystem::exists(cache_filename)) {
        std::cout << "[Main] Found cached ground truth: " << cache_filename << "\n";
        if (load_ground_truth(cache_filename, truth_results, truth_time_ms)) {
            if (truth_results.size() == queries.size()) {
                std::cout << "[Main] Successfully loaded " << truth_results.size() << " cached results.\n";
                loaded = true;
            } else {
                std::cerr << "[Main] Cache size mismatch (Expected " << queries.size() << ", got " << truth_results.size() << "). Regenerating.\n";
                loaded = false;
            }
        } else {
            std::cerr << "[Main] Failed to load cache (corrupted?), re-running BruteForce.\n";
        }
    }

    // 3. If not loaded, run BruteForce and save
    if (!loaded) {
        // Create ground truth and configure (brute)
        auto truth = std::make_unique<BruteForceSearch>();
        truth->configure(args);
        truth->build_index(dataset);

        std::cout << "[Main] Running truth (BruteForce) ...\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        truth_results = run_parallel_search(truth.get(), queries, args.threads, params);
        auto t1 = std::chrono::high_resolution_clock::now();
        double wall_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        std::cout << "[Main] Truth (BruteForce) search completed in " << wall_time_ms / 1000 << " sec\n";
        
        // Recalculate truth_time_ms as sum of individual query times for accurate average
        truth_time_ms = 0.0;
        for (const auto& r : truth_results) {
            truth_time_ms += r.time_ms;
        }

        // Save to cache
        save_ground_truth(cache_filename, truth_results, truth_time_ms);
    } else {
        // Recalculate truth_time_ms from loaded results to ensure it's the sum of latencies
        // (in case the file stored wall-clock time)
        truth_time_ms = 0.0;
        for (const auto& r : truth_results) {
            truth_time_ms += r.time_ms;
        }
    }
    
    if (args.algo == "brute") {
        auto eval = evaluate_results(truth_results, truth_results, args.N, truth_time_ms, truth_time_ms);
        write_results(truth_results, args.output_path, "BruteForce", truth_time_ms, args.config_summary, &truth_results, &eval);
        std::cout << "[Summary] Method=BruteForce\n"
                    << " AF=" << eval.average_AF << "\n"
                    << " Recall@" << args.N << "=" << eval.recall_at_N
                    << " QPS=" << eval.qps << "\n"
                    << " tApproxAvg=" << eval.tApproxAvg << "ms" << "\n"
                    << " tTrueAvg=" << eval.tTrueAvg << "ms\n";
        return 0;
    }

    // Run Given Algorithm (approx)
    std::cout << "[Main] Running approx (" << args.algo << ") ...\n";
    auto ta0 = std::chrono::high_resolution_clock::now();
    auto approx_results = run_parallel_search(approx.get(), queries, args.threads, params);
    auto ta1 = std::chrono::high_resolution_clock::now();
    double approx_wall_ms = std::chrono::duration<double, std::milli>(ta1 - ta0).count();
    std::cout << "[Main] Approx search completed in " << approx_wall_ms / 1000 << " sec\n";

    // Calculate approx_time_ms as sum of individual query times
    double approx_time_ms = 0.0;
    for (const auto& r : approx_results) {
        approx_time_ms += r.time_ms;
    }

    std::cout << "[Main] Mean Approx search " << (approx_results.empty() ? 0.0 : approx_time_ms / approx_results.size() / 1000.0) << " sec\n";

    // Evaluate
    auto eval = evaluate_results(approx_results, truth_results, args.N, approx_time_ms, truth_time_ms);

    // Write approx results
    write_results(approx_results, args.output_path, approx->name(), approx_time_ms, args.config_summary, &truth_results, &eval);

    // Summary output
    std::cout << "[Summary] Method=" << approx->name() << "\n"
              << " AF=" << eval.average_AF << "\n"
              << " Recall@" << args.N << "=" << eval.recall_at_N
              << " QPS=" << eval.qps << "\n"
              << " tApproxAvg=" << eval.tApproxAvg << "ms" << "\n"
              << " tTrueAvg=" << eval.tTrueAvg << "ms\n";
              

    return 0;
}