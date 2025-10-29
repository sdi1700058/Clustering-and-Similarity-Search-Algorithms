
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <numeric>

#include "../../include/utils/data_loader.h"

// Utility function to print sample vectors for verification
static void print_sample_vectors(const std::vector<Vector>& data, int n = 3) {
    std::cout << "[Loader] Preview of first " << n << " vectors:\n";
    for (int i = 0; i < std::min(n, (int)data.size()); ++i) {
        std::cout << "  Vector[" << i << "] = [ ";
        for (int j = 0; j < std::min((int)data[i].values.size(), 10); ++j)
            std::cout << data[i].values[j] << " ";
        if ((int)data[i].values.size() > 10) std::cout << "...";
        std::cout << "] (dim=" << data[i].values.size() << ")\n";
    }
}

namespace data_loader {

// --- MNIST Loader ---
std::vector<Vector> load_mnist(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open MNIST file: " + path);

    uint32_t magic, num_images, rows, cols;
    f.read((char*)&magic, 4); f.read((char*)&num_images, 4);
    f.read((char*)&rows, 4); f.read((char*)&cols, 4);
    magic = __builtin_bswap32(magic);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    std::vector<Vector> out(num_images);
    for (uint32_t i = 0; i < num_images; ++i) {
        out[i].values.resize(rows * cols);
        for (uint32_t j = 0; j < rows * cols; ++j) {
            unsigned char pixel;
            f.read((char*)&pixel, 1);
            out[i].values[j] = static_cast<double>(pixel);
        }
    }
    std::cout << "[MNIST] loaded " << out.size() << " images (" << rows << "x" << cols << ")\n";
    print_sample_vectors(out);
    return out;
}

// --- SIFT Loader ---
std::vector<Vector> load_sift(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open SIFT file: " + path);
    std::vector<Vector> out;
    while (true) {
        int dim;
        if (!f.read((char*)&dim, 4)) break;
        if (dim != 128) break;
        Vector v;
        v.values.resize(dim);
        std::vector<float> buffer(dim);
        f.read(reinterpret_cast<char*>(buffer.data()), dim * sizeof(float));
        if (!f) break;
        for (int idx = 0; idx < dim; ++idx) v.values[idx] = static_cast<double>(buffer[idx]);
        out.push_back(std::move(v));
    }
    std::cout << "[SIFT] loaded " << out.size() << " vectors\n";
    print_sample_vectors(out);
    return out;
}

std::vector<Vector> load_dataset(const std::string& path, const std::string& type) {
    if (type == "mnist") return load_mnist(path);
    if (type == "sift") return load_sift(path);
    throw std::runtime_error("Unknown dataset type: " + type);
}

std::vector<Vector> load_queries(const std::string& path, const std::string& type) {
    return load_dataset(path, type);
}

} // namespace data_loader