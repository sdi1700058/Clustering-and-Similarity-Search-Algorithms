
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cstdint>

#include "../../include/utils/data_loader.h"

/*
 NOTE: CURRENT CHECKPOINT: simple ASCII loader is implemented (format below).
 TODO: Add binary MNIST (big-endian) and SIFT .fvecs (little-endian) loaders
 to conform strictly to the assignment.
*/

// Helper for endian swap
static uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff) |
           ((val << 8) & 0xff0000) |
           ((val >> 8) & 0xff00) |
           ((val << 24) & 0xff000000);
}

// ASCII fallback (for demo/testing)
static std::vector<Vector> load_ascii(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) { std::cerr << "[DataLoader] Cannot open " << path << "\n"; exit(1); }
    int n, d;
    in >> n >> d;
    std::vector<Vector> data(n);
    for (int i = 0; i < n; ++i) {
        data[i].values.resize(d);
        for (int j = 0; j < d; ++j) in >> data[i].values[j];
    }
    std::cout << "[DataLoader] ASCII file loaded (" << n << " x " << d << ")\n";
    return data;
}

// MNIST loader (Big Endian)
static std::vector<Vector> load_mnist(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "[DataLoader] Cannot open MNIST file " << path << "\n"; exit(1); }

    uint32_t magic = 0, num_images = 0, rows = 0, cols = 0;
    f.read((char*)&magic, 4);
    f.read((char*)&num_images, 4);
    f.read((char*)&rows, 4);
    f.read((char*)&cols, 4);
    magic = swap_endian(magic);
    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (magic != 2051) {
        std::cerr << "[DataLoader] Invalid MNIST magic number " << magic << "\n";
        exit(1);
    }

    const uint32_t dim = rows * cols;
    std::vector<Vector> data(num_images);
    for (uint32_t i = 0; i < num_images; ++i) {
        data[i].values.resize(dim);
        for (uint32_t j = 0; j < dim; ++j) {
            unsigned char pixel = 0;
            f.read((char*)&pixel, 1);
            data[i].values[j] = static_cast<float>(pixel);
        }
    }
    std::cout << "[DataLoader] Loaded MNIST dataset (" << num_images << " x " << dim << ")\n";
    return data;
}

// SIFT .fvecs loader (Little Endian)
static std::vector<Vector> load_sift(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "[DataLoader] Cannot open SIFT file " << path << "\n"; exit(1); }

    std::vector<Vector> data;
    while (true) {
        int dim;
        if (!f.read((char*)&dim, 4)) break;
        Vector v;
        v.values.resize(dim);
        f.read(reinterpret_cast<char*>(v.values.data()), dim * 4);
        data.push_back(std::move(v));
    }

    std::cout << "[DataLoader] Loaded SIFT dataset (" << data.size() << " x 128)\n";
    return data;
}

std::vector<Vector> load_vectors(const std::string& path) {
    if (path.find(".fvecs") != std::string::npos)
        return load_sift(path);
    else if (path.find("mnist") != std::string::npos || path.find("MNIST") != std::string::npos)
        return load_mnist(path);
    else
        return load_ascii(path);
}