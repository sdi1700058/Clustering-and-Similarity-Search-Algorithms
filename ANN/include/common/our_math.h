#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace our_math {

    template <typename T>
    inline T mean(const std::vector<T>& vec) {
        if (vec.empty())
            throw std::runtime_error("Cannot compute mean of an empty vector.");

        T sum = std::accumulate(vec.begin(), vec.end(), static_cast<T>(0));
        return sum / static_cast<T>(vec.size());
    }

    inline double mean(const std::vector<double>& vec) {
        return mean<double>(vec);
    }

    template <typename T>
    inline T median(const std::vector<T>& vec) {
        if (vec.empty())
            throw std::runtime_error("Cannot compute median of an empty vector.");

        std::vector<T> current_vector = vec;
        size_t n = current_vector.size();
        size_t median_index = n / 2;

        std::nth_element(
            current_vector.begin(),
            current_vector.begin() + median_index,
            current_vector.end() 
        );

        if (n % 2 != 0) {
            // Odd number of elements
            return current_vector[median_index];
        } else {
            // Even number of elements → average of two middle values
            T val1 = current_vector[median_index];
            T val2 = *std::max_element(current_vector.begin(),
                                       current_vector.begin() + median_index);
            return static_cast<T>((val1 + val2) / static_cast<T>(2.0));
        }
    }

    inline double median(const std::vector<double>& vec) {
        return median<double>(vec);
    }

} // namespace our_math

#endif // MATH_UTILS_H
