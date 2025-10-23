#ifndef ARGS_PARSER_H
#define ARGS_PARSER_H

/*
    This is the header file for the command-line arguments parser.
    It defines the Args structure and the parse_args function.
    
    Add new arguments as needed, please document them here.
    Current Arguments:
        --dataset_path : Path to the dataset file
        --query_path   : Path to the query file
        --output_path  : Path to the output file
        --type         : Type of algorithm to use (e.g., "dummy")
        --threads      : Number of threads to use
        --N            : Number of neighbors to search for
        --R            : Search radius
        --interactive  : Whether to run in interactive mode

    Currently Implemented Algorithms:
        - Dummy Search Algorithm
*/

#include <string>
#include <unordered_map>

struct Args {
    std::string dataset_path;
    std::string query_path;
    std::string output_path;
    std::string type;
    int threads = 4;
    int N = 5;
    int R = 100;
    bool interactive = true;
};

Args parse_args(int argc, char** argv);

#endif