#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <cstdint>
#include <cstring>

#include "../include/utils/args_parser.h"
#include "../include/algorithms/brute_force_search.h"
#include "../include/algorithms/dummy_search.h"
#include "../include/algorithms/search_algorithm.h"
#include "../include/utils/algorithm_factory.h"
#include "../include/utils/parallel_runner.h"
#include "../include/utils/data_loader.h"
#include "../include/utils/result_writer.h"
#include "../include/utils/truth_cahce.h"
#include "../include/common/metrics.h"
#include "../include/common/evaluation_metrics.h"

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

    // Create ground truth and configure (brute)
    //auto truth = std::make_unique<BruteForceSearch>();
    //truth->configure(args);
    //truth->build_index(dataset);

    // Run Ground Truth (brute)
    //std::cout << "[Main] Running truth (BruteForce) ...\n";
    //auto t0 = std::chrono::high_resolution_clock::now();
    //auto truth_results = run_parallel_search(truth.get(), queries, args.threads, params);
    //auto t1 = std::chrono::high_resolution_clock::now();
    //double truth_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Run Given Algorithm (approx)
    std::cout << "[Main] Running approx (" << args.algo << ") ...\n";
    auto ta0 = std::chrono::high_resolution_clock::now();
    auto approx_results = run_parallel_search(approx.get(), queries, args.threads, params);
    auto ta1 = std::chrono::high_resolution_clock::now();
    double approx_time_ms = std::chrono::duration<double, std::milli>(ta1 - ta0).count();
    std::cout << "[Main] Approx search completed in " << approx_time_ms << " ms\n";

    std::vector<SearchResult> truth_results;
    double truth_time_ms = 0.0;
    EvalResults eval_summary;
    const EvalResults* eval_ptr = nullptr;
    const std::vector<SearchResult>* truth_ptr = nullptr;

    if (args.eval) {
        uint64_t dataset_hash = truth_cache::hash_vector_list(dataset);
        uint64_t query_hash = truth_cache::hash_vector_list(queries);
        auto cache_path = truth_cache::cache_path_for(args, dataset_hash, query_hash);

        bool truth_ready = false;
        if (truth_cache::load(cache_path, truth_results, truth_time_ms)) {
            truth_ready = true;
        } else if (args.algo == "brute") {
            truth_results = approx_results;
            truth_time_ms = approx_time_ms;
            truth_ready = true;
            truth_cache::save(cache_path, truth_results, truth_time_ms);
        } else {
            std::cout << "[Main] Running truth (BruteForce) ...\n";
            auto truth = std::make_unique<BruteForceSearch>();
            truth->configure(args);
            truth->build_index(dataset);
            auto t0 = std::chrono::high_resolution_clock::now();
            truth_results = run_parallel_search(truth.get(), queries, args.threads, params);
            auto t1 = std::chrono::high_resolution_clock::now();
            truth_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "[Main] Truth search completed in " << truth_time_ms << " ms\n";
            truth_cache::save(cache_path, truth_results, truth_time_ms);
            truth_ready = true;
        }

        if (truth_ready) {
            eval_summary = evaluate_results(approx_results, truth_results, args.N, approx_time_ms, truth_time_ms);
            eval_ptr = &eval_summary;
            truth_ptr = &truth_results;
        }
    } else {
        std::cout << "[Main] Evaluation disabled (-eval false); skipping ground truth.\n";
    }

    std::cout << "[Main] Writing results to " << args.output_path << " ...\n";
    write_results(approx_results, args.output_path, approx->name(), approx_time_ms, args.config_summary, truth_ptr, eval_ptr);

    return 0;
}