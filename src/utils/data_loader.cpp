#include <fstream>
#include <iostream>

#include "../../include/utils/data_loader.h"

std::vector<Vector> load_vectors(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << path << std::endl;
        exit(1);
    }

    int n, d;
    file >> n >> d;
    std::vector<Vector> vectors(n);

    for (int i = 0; i < n; ++i) {
        vectors[i].values.resize(d);
        for (int j = 0; j < d; ++j)
            file >> vectors[i].values[j];
    }

    std::cout << "[DataLoader] Loaded " << n << " vectors from " << path << ".\n";
    return vectors;
}