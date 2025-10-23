#include <fstream>
#include <iostream>

#include "../../include/utils/result_writer.h"

void write_results(const std::vector<SearchResult>& results, const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "[ERROR] Cannot open output file: " << path << std::endl;
        return;
    }

    for (const auto& r : results) {
        out << "Query_id: " << r.query_id << "\n";
        out << "Neighbors: ";
        for (auto id : r.neighbor_ids) out << id << " ";
        out << "\nDistances: ";
        for (auto d : r.distances) out << d << " ";
        out << "\nTime_ms: " << r.time_ms << "\n";
        out << "----------------------------------------\n";
    }
    std::cout << "[Writer] Results written to " << path << std::endl;
}