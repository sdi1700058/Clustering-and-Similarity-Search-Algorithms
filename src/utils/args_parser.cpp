#include <iostream>
#include <sstream>

#include "../../include/utils/args_parser.h"

Args parse_args(int argc, char** argv) {
    Args args;
    std::unordered_map<std::string, std::string> map;

    for (int i = 1; i < argc - 1; i += 2)
        map[argv[i]] = argv[i + 1];

    auto get_or_prompt = [&](const std::string& key, const std::string& message) -> std::string {
        if (map.count(key)) return map[key];
        if (!args.interactive) return "";
        std::string val;
        while (true) {
            std::cout << "[INPUT] " << message << " (q to quit): ";
            std::getline(std::cin, val);
            if (val == "q" || val == "Q") {
                std::cout << "Exiting...\n";
                exit(0);
            }
            if (!val.empty()) break;
        }
        return val;
    };

    args.dataset_path = get_or_prompt("-d", "Enter dataset path");
    args.query_path = get_or_prompt("-q", "Enter query path");
    args.output_path = get_or_prompt("-o", "Enter output path");
    args.type = get_or_prompt("-type", "Enter dataset type (mnist/sift)");
    args.threads = map.count("-threads") ? std::stoi(map["-threads"]) : 4;
    args.N = map.count("-N") ? std::stoi(map["-N"]) : 5;
    args.R = map.count("-R") ? std::stoi(map["-R"]) : 100;

    std::cout << "\n[INFO] Using configuration:\n"
              << "  Dataset: " << args.dataset_path << "\n"
              << "  Queries: " << args.query_path << "\n"
              << "  Output: " << args.output_path << "\n"
              << "  Type: " << args.type << "\n"
              << "  Threads: " << args.threads << "\n"
              << "  N: " << args.N << ", R: " << args.R << "\n";

    return args;
}