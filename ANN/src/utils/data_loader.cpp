
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

// --- SIFT/FVECS Loader (Generic) ---
std::vector<Vector> load_sift(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open FVECS file: " + path);
    
    // Check file size to avoid infinite loops on empty files
    f.seekg(0, std::ios::end);
    if (f.tellg() == 0) return {};
    f.seekg(0, std::ios::beg);

    std::vector<Vector> out;
    int expected_dim = -1;

    while (f.peek() != EOF) {
        int dim;
        // Read dimension (4 bytes int)
        if (!f.read((char*)&dim, 4)) break;
        
        // Sanity check for dimension (e.g., must be between 1 and 20000)
        if (dim <= 0 || dim > 20000) {
             std::cerr << "[Loader] Warning: Encountered invalid dimension " << dim << " or EOF.\n";
             break;
        }

        if (expected_dim == -1) expected_dim = dim;
        else if (dim != expected_dim) {
            std::cerr << "[Loader] Error: Inconsistent vector dimensions. Expected " << expected_dim << ", got " << dim << "\n";
            // Don't break immediately if you want partial load, but usually we break or skip
            break;
        }

        Vector v;
        v.values.resize(dim);
        std::vector<float> buffer(dim);
        
        // Read floats
        if (!f.read(reinterpret_cast<char*>(buffer.data()), dim * sizeof(float))) break;
        
        for (int idx = 0; idx < dim; ++idx) {
            v.values[idx] = static_cast<double>(buffer[idx]);
        }
        out.push_back(std::move(v));
    }
    std::cout << "[FVECS] loaded " << out.size() << " vectors of dim " << expected_dim << "\n";
    // print_sample_vectors(out);
    return out;
}

std::vector<Vector> load_dataset(const std::string& path, const std::string& type) {
    if (type == "mnist") return load_mnist(path);
    if (type == "sift") return load_sift(path);
    if (type == "protein") return load_sift(path);
    throw std::runtime_error("Unknown dataset type: " + type);
}

std::vector<Vector> load_queries(const std::string& path, const std::string& type) {
    return load_dataset(path, type);
}

} // namespace data_loader