#include <iostream>

#include "../include/utils/args_parser.h"
#include "../include/algorithms/dummy_search.h"
#include "../include/algorithms/search_algorithm.h"
#include "../include/utils/parallel_runner.h"
#include "../include/utils/data_loader.h"
#include "../include/utils/result_writer.h"

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
    std::cout << "=== ANN Search Framework (Dummy Build) ===\n";

    Args args = parse_args(argc, argv);
    Params params{args.N, args.R};

    // Load data
    auto dataset = load_vectors(args.dataset_path);
    auto queries = load_vectors(args.query_path);

    // Create dummy algorithm
    DummySearch algo;
    algo.build_index(dataset);

    // Run parallel dummy search
    auto results = run_parallel_search(&algo, queries, args.threads, params);

    // Write output
    write_results(results, args.output_path);

    std::cout << "\n[Done] All queries processed successfully.\n";
    return 0;
}
