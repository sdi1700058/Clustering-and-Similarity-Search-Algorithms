#ifndef ARGS_PARSER_H
#define ARGS_PARSER_H

/*
    This is the header file for the command-line arguments parser.
    It defines the Args structure and the parse_args function.
    
    Add new arguments as needed, please document them here.
    Current Arguments:
        - Dataset Path (-d): Path to the dataset file.
        - Query Path (-q): Path to the query file.
        - Output Path (-o): Path to save the results.
        - Dataset Type (-type): Type of dataset (demo/mnist/sift).
        - Algorithm (-algo): Algorithm to use (brute/dummy).
        - Distance Metric (-metric): Distance metric to use (l1/l2).
        - Threads (-threads): Number of threads for parallel execution.
        - N (-N): Number of nearest neighbors to search for.
        - R (-R): Search radius for range queries.
    Currently Implemented Algorithms:
        - Brute-Force Search Algorithm (used as ground truth)
        - Dummy Search Algorithm (check parallel execution)
    Currently Implemented Distance Metrics:
        - L1 (Manhattan) Distance
        - L2 (Euclidean) Distance
*/

#pragma once
#include <string>
#include <unordered_map>
#include <iostream>

struct Args {
    std::string dataset_path, query_path, output_path;
    std::string type, algo, metric;
    int threads, N;
    double R;
    bool range = true;
    bool interactive = true;
    std::string config_summary;

    // Algorithm-specific params
    int seed = 1;
    int k = 4, L = 5;             // LSH
    double w = 4.0;
    int kproj = 14, M = 10, probes = 2; // Hypercube
    int kclusters = 50, nprobe = 5;     // IVFFlat / IVFPQ
    int pq_M = 16, pq_nbits = 8;        // IVFPQ
};

Args parse_args(int argc, char** argv);

#endif // ARGS_PARSER_H