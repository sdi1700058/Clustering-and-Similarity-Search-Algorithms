#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>

#include "../include/utils/args_parser.h"
#include "../include/algorithms/brute_force_search.h"
#include "../include/algorithms/dummy_search.h"
#include "../include/algorithms/search_algorithm.h"
#include "../include/utils/parallel_runner.h"
#include "../include/utils/data_loader.h"
#include "../include/utils/result_writer.h"
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
    std::cout << "=== ANN Search Framework ===\n";

    // Parse CLI arguments
    Args args = parse_args(argc, argv);
    Params params{args.N, args.R};

    // Metric configuration
    metrics::MetricConfig metric_cfg = metrics::parse_metric_type(args.metric);
    BruteForceSearch::set_metric_config(metric_cfg);

    // Load dataset & queries
    auto dataset = load_vectors(args.dataset_path);
    auto queries = load_vectors(args.query_path);

    // Select algorithm
    std::unique_ptr<SearchAlgorithm> algo;
    if (args.algo == "dummy") {
        algo = std::make_unique<DummySearch>();
    } else if (args.algo == "brute") {
        algo = std::make_unique<BruteForceSearch>();
    } else {
        std::cerr << "[ERROR] Unknown algorithm: " << args.algo << "\n";
        return 1;
    }

    // Build index
    algo->build_index(dataset);

    // Run parallel search
    auto start_time = std::chrono::high_resolution_clock::now();
    auto results = run_parallel_search(algo.get(), queries, args.threads, params);
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Optional: compute “ground truth” using brute-force (for evaluation/demo)
    auto truth_algo = std::make_unique<BruteForceSearch>();
    truth_algo->build_index(dataset);
    auto truth_results = run_parallel_search(truth_algo.get(), queries, args.threads, params);

    // Evaluate
    EvalResults eval = evaluate_results(results, truth_results, params.N, total_time_ms);
    std::cout << "[Main] Total search time: " << total_time_ms << " ms\n";
    // Write output
    write_results(results, args.output_path, args.algo);
    // Prints Results to console
    std::cout << "=== Output ===\n";
    std::ifstream result_file(args.output_path);
    std::cout << result_file.rdbuf();
    std::cout << "=== End of Output ===\n";

    std::cout << "\n[Done] All queries processed successfully.\n";
    return 0;
}