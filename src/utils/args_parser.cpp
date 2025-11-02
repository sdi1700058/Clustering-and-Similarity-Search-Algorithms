#include <iostream>
#include <unordered_map>
#include <sstream>
#include <cstdlib>
#include <algorithm>

#include "../../include/utils/args_parser.h"

Args parse_args(int argc, char** argv) {
    Args args;
    std::unordered_map<std::string,std::string> mp;

    for (int i = 1; i + 1 < argc; i += 2) {
        mp[argv[i]] = argv[i+1];
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

    // --- General Options ---
    // Paths
    args.dataset_path = get_or_prompt("-d", "Enter dataset path", "data/sample_input.dat");
    args.query_path   = get_or_prompt("-q", "Enter query path", "data/sample_queries.dat");
    args.output_path  = get_or_prompt("-o", "Enter output path", "output/results.txt");
    // Common parameters
    args.type         = mp.count("-type") ? mp["-type"] : // dataset type
                        get_or_prompt("-type", "Enter dataset type (demo/mnist/sift)", "demo");
    args.algo         = mp.count("-algo") ? mp["-algo"] : // algorithm to use
                        get_or_prompt("-algo", "Enter algorithm (brute/dummy/lsh/hypercube/ivfflat/ivfpq)", "dummy");
    args.metric       = mp.count("-metric") ? mp["-metric"] : // distance METRIC
                        get_or_prompt("-metric", "Enter distance metric (l1/l2)", "l2");
    args.threads      = mp.count("-threads") ? std::stoi(mp["-threads"]) :
                        std::stoi(get_or_prompt("-threads", "Enter number of threads", "1"));
    args.N            = mp.count("-N") ? std::stoi(mp["-N"]) : // NUMber of NEAREST neighbors
                        std::stoi(get_or_prompt("-N", "Enter number of nearest neighbors N", "1"));
    args.R            = mp.count("-R") ? std::stod(mp["-R"]) : // search Radius
                        std::stod(get_or_prompt("-R", "Enter search radius R", "0.0"));
    if (mp.count("-range")) {
        std::string v = mp["-range"];
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        args.range = (v == "true" || v == "1" || v == "yes");
    } else {
        std::string def = "true";
        std::string val = get_or_prompt("-range", "Enable range search? (true/false)", def);
        std::string tmp = val;
        std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
        args.range = (tmp == "true" || tmp == "1" || tmp == "yes");
    }

    // --- Algorithm-specific interactive options ---
    /* *** LSH Specific Parameters *** */
    if (args.algo == "lsh") {
        args.seed = std::stoi(get_or_prompt("-seed", "Enter seed", "1"));
        args.k = std::stoi(get_or_prompt("-k", "Enter number of hash functions k", "4"));
        args.L = std::stoi(get_or_prompt("-L", "Enter number of hash tables L", "5"));
        args.w = std::stod(get_or_prompt("-w", "Enter window size w", "4.0"));
    } 
    /* *** Hypercube Specific Parameters *** */
    else if (args.algo == "hypercube") {
        args.seed = std::stoi(get_or_prompt("-seed", "Enter seed", "1"));
        args.kproj = std::stoi(get_or_prompt("-kproj", "Enter projection dimension (d')", "14"));
        args.M = std::stoi(get_or_prompt("-M", "Enter max candidate points M", "10"));
        args.probes = std::stoi(get_or_prompt("-probes", "Enter max probes", "2"));
        args.w = std::stod(get_or_prompt("-w", "Enter window size w", "4.0"));
    } 
    /* *** IVFFlat Specific Parameters *** */
    else if (args.algo == "ivfflat") {
        args.seed = std::stoi(get_or_prompt("-seed", "Enter seed", "1"));
        args.kclusters = std::stoi(get_or_prompt("-kclusters", "Enter number of clusters k", "50"));
        args.nprobe = std::stoi(get_or_prompt("-nprobe", "Enter clusters to probe", "5"));
    }
    /* *** IVFPQ Specific Parameters *** */
    else if (args.algo == "ivfpq") {
        args.seed = std::stoi(get_or_prompt("-seed", "Enter seed", "1"));
        args.kclusters = std::stoi(get_or_prompt("-kclusters", "Enter number of clusters k", "50"));
        args.nprobe = std::stoi(get_or_prompt("-nprobe", "Enter clusters to probe", "5"));
        args.pq_M = std::stoi(get_or_prompt("-M", "Enter number of sub-vectors M", "16"));
        args.pq_nbits = std::stoi(get_or_prompt("-nbits", "Enter nbits (2^nbits clusters)", "8"));
    }

    // Print the final configuration
    if (args.algo == "brute" || args.algo == "dummy") {
        // Config without algorithm-specific parameters
        std::ostringstream info;
        info << "\n[INFO] Using configuration:\n"
                << "  Dataset: " << args.dataset_path << "\n"
                << "  Queries: " << args.query_path << "\n"
                << "  Output: " << args.output_path << "\n"
                << "  Type: " << args.type << "\n"
                << "  Algorithm: " << args.algo << "\n"
                << "  Metric: " << args.metric << "\n"
                << "  Threads: " << args.threads << "\n"
                << "  N=" << args.N << " R=" << args.R
                << " Range=" << (args.range ? "truee" : "false") << "\n";
        args.config_summary = info.str();
        std::cout << args.config_summary;
    } else if (args.algo == "lsh") {
        std::ostringstream info;
        info << "\n[INFO] Using configuration:\n"
                << "  Dataset: " << args.dataset_path << "\n"
                << "  Queries: " << args.query_path << "\n"
                << "  Output: " << args.output_path << "\n"
                << "  Type: " << args.type << "\n"
                << "  Algorithm: " << args.algo << "\n"
                << "  Metric: " << args.metric << "\n"
                << "  Threads: " << args.threads << "\n"
                << "  N=" << args.N << " R=" << args.R
                << " Range=" << (args.range ? "true" : "false")
                << "  Seed=" << args.seed << " k=" << args.k << " L=" << args.L << " w=" << args.w << "\n";
    } else if (args.algo == "hypercube") {
        std::ostringstream info;
        info << "\n[INFO] Using configuration:\n"
                << "  Dataset: " << args.dataset_path << "\n"
                << "  Queries: " << args.query_path << "\n"
                << "  Output: " << args.output_path << "\n"
                << "  Type: " << args.type << "\n"
                << "  Algorithm: " << args.algo << "\n"
                << "  Metric: " << args.metric << "\n"
                << "  Threads: " << args.threads << "\n"
                << "  N=" << args.N << " R=" << args.R
                << " Range=" << (args.range ? "true" : "false")
                << "  Seed=" << args.seed << " kproj=" << args.kproj << " M=" << args.M
                <<" probes="<< args.probes <<" w="<< args.w<<"\n";
        args.config_summary = info.str();
        std::cout << args.config_summary;
    } else if (args.algo == "ivfflat" || args.algo == "ivfpq") {
        std::ostringstream info;
        info << "\n[INFO] Using configuration:\n"
                << "  Dataset: " << args.dataset_path << "\n"
                << "  Queries: " << args.query_path << "\n"
                << "  Output: " << args.output_path << "\n"
                << "  Type: " << args.type << "\n"
                <<"  Algorithm: "<< args.algo<<"\n"
                <<"  Metric: "<< args.metric<<"\n"
                <<"  Threads: "<< args.threads<<"\n"
                <<"  N="<< args.N<<" R="<< args.R<<" Range=" << (args.range ? "true" : "false") <<"\n"
                <<"  Seed="<< args.seed<<" kclusters="<< args.kclusters<<" nprobe="<< args.nprobe;
        if (args.algo == "ivfpq") {
            info << " M=" << args.pq_M << " nbits=" << args.pq_nbits << "\n";
        } else {
            info << "\n";
        }
        args.config_summary = info.str();
        std::cout << args.config_summary;
    }
    return args;
}