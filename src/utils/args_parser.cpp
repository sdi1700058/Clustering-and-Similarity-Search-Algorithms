#include <iostream>
#include <unordered_map>
#include <sstream>

#include "../../include/utils/args_parser.h"

Args parse_args(int argc, char** argv) {
    Args args;
    std::unordered_map<std::string,std::string> mp;

    for (int i = 1; i + 1 < argc; i += 2) {
        std::string key = argv[i];
        std::string val = argv[i+1];
        mp[key] = val;
    }

    auto get_or_prompt = [&](const std::string& key, const std::string& message, const std::string& def="") -> std::string {
        if (mp.count(key)) return mp[key];
        if (!args.interactive) return def;
        std::string val;
        while (true) {
            std::cout << "[INPUT] " << message;
            if (!def.empty()) std::cout << " (default: " << def << ")";
            std::cout << " (q to quit): ";
            std::getline(std::cin, val);
            if (val == "q" || val == "Q") { std::cout << "Exiting...\n"; exit(0); }
            if (!val.empty()) break;
            if (!def.empty()) { val = def; break; }
        }
        return val;
    };

    args.dataset_path = get_or_prompt("-d", "Enter dataset path", "data/sample_input.dat");
    args.query_path   = get_or_prompt("-q", "Enter query path", "data/sample_queries.dat");
    args.output_path  = get_or_prompt("-o", "Enter output path", "output/results.txt");
    args.type         = mp.count("-type") ? mp["-type"] : get_or_prompt("-type", "Enter dataset type (demo/mnist/sift)", "demo");
    args.algo         = mp.count("-algo") ? mp["-algo"] : get_or_prompt("-algo", "Enter algorithm (brute/dummy)", "brute");
    args.metric       = mp.count("-metric") ? mp["-metric"] : get_or_prompt("-metric", "Enter distance metric (l1/l2/lpX)", "l2");
    args.threads      = mp.count("-threads") ? std::stoi(mp["-threads"]) : 4;
    args.N            = mp.count("-N") ? std::stoi(mp["-N"]) : 1;
    args.R            = mp.count("-R") ? std::stod(mp["-R"]) : 0.0;

    std::cout << "\n[INFO] Using configuration:\n"
                << "  Dataset (-d): " << args.dataset_path << "\n"
                << "  Queries (-q): " << args.query_path << "\n"
                << "  Output (-o): " << args.output_path << "\n"
                << "  Type (-type): " << args.type << "\n"
                << "  Algorithm (-algo): " << args.algo << "\n"
                << "  Metric (-metric): " << args.metric << "\n"
                << "  Threads (-threads): " << args.threads << "\n"
                << "  N (-N): " << args.N << ", R (-R): " << args.R << "\n";
    return args;
}