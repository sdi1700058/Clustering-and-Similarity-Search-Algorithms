#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_set>

#include "../../include/algorithms/ivfpq_search.h"
#include "../../include/utils/args_parser.h"
#include "../../include/common/metrics.h"
#include "../../include/common/our_math.h"

namespace {
using Clock = std::chrono::high_resolution_clock;

double squared_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

Vector make_zero_vector(int dim) {
    Vector v;
    v.values.assign(dim, 0.0);
    return v;
}

std::vector<Vector> kmeans_pp(const std::vector<Vector>& points,
                              int desired_k,
                              int dim,
                              std::mt19937& rng) {
    if (points.empty()) {
        std::vector<Vector> zeros(desired_k, make_zero_vector(dim));
        return zeros;
    }

    int actual_k = std::min(desired_k, static_cast<int>(points.size()));
    std::uniform_int_distribution<int> pick_first(0, static_cast<int>(points.size()) - 1);

    std::vector<Vector> centers;
    centers.reserve(actual_k);
    centers.push_back(points[static_cast<size_t>(pick_first(rng))]);

    std::vector<double> min_dist(points.size(), std::numeric_limits<double>::infinity());

    while (static_cast<int>(centers.size()) < actual_k) {
        double total = 0.0;
        for (size_t i = 0; i < points.size(); ++i) {
            double d = squared_distance(points[i].values, centers.back().values);
            if (d < min_dist[i]) min_dist[i] = d;
            total += min_dist[i];
        }
        if (total == 0.0) break;

        std::uniform_real_distribution<double> dist(0.0, total);
        double r = dist(rng);
        double cumulative = 0.0;
        for (size_t i = 0; i < points.size(); ++i) {
            cumulative += min_dist[i];
            if (cumulative >= r) {
                centers.push_back(points[i]);
                break;
            }
        }
    }

    centers.resize(actual_k);

    const int max_iters = 5;
    std::vector<int> assignment(points.size(), -1);

    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;
        for (size_t i = 0; i < points.size(); ++i) {
            double best = std::numeric_limits<double>::infinity();
            int best_idx = -1;
            for (int c = 0; c < actual_k; ++c) {
                double d = squared_distance(points[i].values, centers[static_cast<size_t>(c)].values);
                if (d < best) {
                    best = d;
                    best_idx = c;
                }
            }
            if (assignment[i] != best_idx) {
                assignment[i] = best_idx;
                changed = true;
            }
        }
        if (!changed) break;

        std::vector<std::vector<double>> accum(actual_k, std::vector<double>(dim, 0.0));
        std::vector<int> counts(actual_k, 0);
        for (size_t i = 0; i < points.size(); ++i) {
            int cid = assignment[i];
            ++counts[cid];
            for (int d = 0; d < dim; ++d) {
                accum[cid][d] += points[i].values[d];
            }
        }

        for (int c = 0; c < actual_k; ++c) {
            if (counts[c] == 0) {
                std::uniform_int_distribution<int> pick(0, static_cast<int>(points.size()) - 1);
                centers[static_cast<size_t>(c)] = points[static_cast<size_t>(pick(rng))];
                continue;
            }
            for (int d = 0; d < dim; ++d) {
                centers[static_cast<size_t>(c)].values[d] = accum[c][d] / static_cast<double>(counts[c]);
            }
        }
    }

    if (actual_k < desired_k) {
        centers.resize(desired_k);
        for (int i = actual_k; i < desired_k; ++i) {
            centers[static_cast<size_t>(i)] = centers[static_cast<size_t>(i % actual_k)];
        }
    }

    return centers;
}
} // namespace

void IVFPQSearch::configure(const Args& args) {
    p.seed = args.seed;
    p.kclusters = args.kclusters;
    p.nprobe = args.nprobe;
    p.M = args.pq_M;
    p.nbits = args.pq_nbits;
    p.N = args.N;
    p.R = args.R;
    rng.seed(static_cast<std::mt19937::result_type>(p.seed));
}

void IVFPQSearch::build_index(const std::vector<Vector>& dataset) {
    data = dataset;
    n_points_ = static_cast<int>(data.size());
    index_built = false;

    if (data.empty()) {
        std::cout << "[IVFPQ] dataset is empty, nothing to index\n";
        return;
    }

    space_dim_ = static_cast<int>(data.front().values.size());
    if (space_dim_ == 0) {
        throw std::runtime_error("[IVFPQ] dataset vectors have zero dimension");
    }
    if (p.M <= 0) {
        throw std::runtime_error("[IVFPQ] sub-vector count M must be positive");
    }
    if (space_dim_ % p.M != 0) {
        throw std::runtime_error("[IVFPQ] vector dimension must be divisible by M");
    }
    subvector_dim_ = space_dim_ / p.M;

    codebook_size_ = 1 << p.nbits;
    if (codebook_size_ <= 0) {
        throw std::runtime_error("[IVFPQ] invalid codebook size");
    }

    // 1.Lloyd's Clustering
    random_subset();
    if (subset_data.empty()) {
        subset_data = data;
        subset_assignments_.assign(subset_data.size(), 0);
    }

    // Step 1.a: Initialize the centroids using init++
    initialization();

    // set manually a threshold for stoping the algorithm
    int threshold = 1;
    // number of changed vector. It is going to change in each itteration
    int changed = std::numeric_limits<int>::max();
    int guard = 0;
    // repeat until the threshold is not reached
    while (changed > threshold && guard < 10) {
        // Step 1.b: call the appropriate method of assignment
        changed = assignment_lloyds();
        std::cout << "[IVFPQ] Assignment completed with " << changed << " changed vectors.\n";
        // Step 1.c: Update the centroids
        update();
        ++guard;
    }
    std::cout << "[IVFPQ] Coarse clustering completed in " << guard
              << " iterations with " << p.kclusters << " clusters.\n";
    // 2. Build Inverted Lists
    data_assignments_.assign(n_points_, -1);
    for (int i = 0; i < n_points_; ++i) {
        // 2.Assign to nearest centroid
        data_assignments_[i] = nearest_centroid(data[static_cast<size_t>(i)]);
    }
    std::cout << "[IVFPQ] Data assignment to centroids completed.\n";
    build_pq_codebooks();
    std::cout << "[IVFPQ] PQ codebooks built with " << codebook_size_
              << " centroids per sub-vector.\n";
    // 3. Encode points and build inverted lists
    point_codes_.assign(n_points_, std::vector<std::uint8_t>(p.M, 0));
    inverted_lists_.assign(static_cast<size_t>(p.kclusters), {});
    for (int i = 0; i < n_points_; ++i) {
        int cid = data_assignments_[i];
        if (cid < 0) continue;
        point_codes_[static_cast<size_t>(i)] = encode_point(data[static_cast<size_t>(i)], cid);
        // 3.Append to Inverted List
        inverted_lists_[static_cast<size_t>(cid)].push_back(i);
    }
    std::cout << "[IVFPQ] Inverted lists built with " << inverted_lists_.size() << " clusters.\n";
    index_built = true;
    std::cout << "[IVFPQ] index built with " << data.size()
              << " vectors (dim=" << space_dim_
              << ", k=" << p.kclusters
              << ", M=" << p.M
              << ", codebook=" << codebook_size_ << ")\n";
}

SearchResult IVFPQSearch::search(const Vector& query, const Params& params, int query_id) const {
    auto t0 = Clock::now();
    SearchResult res;
    res.query_id = query_id;

    if (!index_built || data.empty() || centroids.empty()) {
        res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return res;
    }
    std::cout << "[IVFPQ] Search started for query ID " << query_id << ".\n";
    if (static_cast<int>(query.values.size()) != space_dim_) {
        res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return res;
    }

    // 1. Distance to all centroids & select top 'nprobes'
    // a. Calculate distance to all k centroids
    std::vector<std::pair<int, double>> coarse;
    coarse.reserve(static_cast<size_t>(p.kclusters));
    for (int j = 0; j < static_cast<int>(centroids.size()); ++j) {
        double dist = metrics::distance(query.values, centroids[static_cast<size_t>(j)].values, metrics::GLOBAL_METRIC_CFG);
        coarse.emplace_back(j, dist);
    }

    // b. Select the top 'nprobes' closest centroids
    size_t effective_nprobe = std::min(static_cast<size_t>(p.nprobe), coarse.size());
    if (effective_nprobe == 0) effective_nprobe = std::min<size_t>(1, coarse.size());
    std::nth_element(coarse.begin(), coarse.begin() + effective_nprobe, coarse.end(),
                     [](const auto& a, const auto& b) { return a.second < b.second; });
    coarse.resize(effective_nprobe);

    // Step 2: compute residual and LUT values for PQ
    struct Candidate { int idx; double dist; };
    std::vector<Candidate> candidates;
    candidates.reserve(256);

    std::unordered_set<int> visited;

    std::vector<std::vector<double>> lut(static_cast<size_t>(p.M), std::vector<double>(static_cast<size_t>(codebook_size_), 0.0));

    for (const auto& entry : coarse) {
        int cid = entry.first;
        if (cid < 0 || cid >= static_cast<int>(centroids.size())) continue;

        std::vector<double> residual(space_dim_);
        const auto& centroid = centroids[static_cast<size_t>(cid)].values;
        for (int d = 0; d < space_dim_; ++d) {
            residual[static_cast<size_t>(d)] = query.values[static_cast<size_t>(d)] - centroid[static_cast<size_t>(d)];
        }

        for (int m = 0; m < p.M; ++m) {
            size_t offset = static_cast<size_t>(m * subvector_dim_);
            for (int h = 0; h < codebook_size_; ++h) {
                const auto& code_centroid = pq_codebooks_[static_cast<size_t>(m)][static_cast<size_t>(h)].values;
                double accum = 0.0;
                for (int d = 0; d < subvector_dim_; ++d) {
                    double diff = residual[offset + static_cast<size_t>(d)] - code_centroid[static_cast<size_t>(d)];
                    accum += diff * diff;
                }
                lut[static_cast<size_t>(m)][static_cast<size_t>(h)] = accum;
            }
        }

        for (int idx : inverted_lists_[static_cast<size_t>(cid)]) {
            if (!visited.insert(idx).second) continue;

            const auto& codes = point_codes_[static_cast<size_t>(idx)];
            if (static_cast<int>(codes.size()) != p.M) continue;

            // Step 3: accumulate ADC distance
            double dist_sq = 0.0;
            for (int m = 0; m < p.M; ++m) {
                std::size_t code = static_cast<std::size_t>(codes[static_cast<size_t>(m)]);
                dist_sq += lut[static_cast<size_t>(m)][code];
            }
            double dist = std::sqrt(dist_sq);
            candidates.push_back({idx, dist});
        }
    }

    if (candidates.empty()) {
        for (size_t i = 0; i < data.size(); ++i) {
            double dist = metrics::distance(query.values, data[i].values, metrics::GLOBAL_METRIC_CFG);
            candidates.push_back({static_cast<int>(i), dist});
        }
    }

    // Step 4: keep top-N and range results
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.dist < b.dist; });

    int topK = std::min(params.N, static_cast<int>(candidates.size()));
    for (int i = 0; i < topK; ++i) {
        res.neighbor_ids.push_back(candidates[static_cast<size_t>(i)].idx);
        res.distances.push_back(static_cast<float>(candidates[static_cast<size_t>(i)].dist));
    }

    if (params.enable_range && params.R > 0.0) {
        for (const auto& cand : candidates) {
            if (cand.dist <= params.R) {
                res.range_neighbor_ids.push_back(cand.idx);
                res.range_distances.push_back(static_cast<float>(cand.dist));
            } else {
                break;
            }
        }
    }
    std::cout << "[IVFPQ] Search completed for query ID " << query_id << ".\n";
    auto t1 = Clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

// Utility: centroid accessors -------------------------------------------------

std::vector<std::vector<int>> IVFPQSearch::get_centroids_map() const {
    // we want to keep a map containing the true values of the vectors, in order to find the median
    std::vector<std::vector<int>> map(static_cast<size_t>(p.kclusters));
    // insert the vectors in the map
    for (int i = 0; i < n_points_; ++i) {
        int cid = (i < static_cast<int>(data_assignments_.size())) ? data_assignments_[i] : -1;
        if (cid >= 0 && cid < p.kclusters) {
            map[static_cast<size_t>(cid)].push_back(i);
        }
    }
    return map;
}

// Silhouette scores ----------------------------------------------------------

std::pair<std::vector<double>, double> IVFPQSearch::compute_silhouette_fast() const {
    std::vector<double> silhouettes(static_cast<size_t>(p.kclusters), 0.0);
    std::vector<int> assigned(static_cast<size_t>(p.kclusters), 0);
    double total_sil = 0.0;

    for (int i = 0; i < n_points_; ++i) {
        int nearest = (i < static_cast<int>(data_assignments_.size())) ? data_assignments_[i] : -1;
        if (nearest < 0) continue;

        // a_i: distance to own centroid
        double a_i = metrics::distance(data[static_cast<size_t>(i)].values,
                                       centroids[static_cast<size_t>(nearest)].values,
                                       metrics::GLOBAL_METRIC_CFG);

        // b_i: distance to nearest other centroid
        double b_i = std::numeric_limits<double>::max();
        for (int c = 0; c < p.kclusters; ++c) {
            if (c == nearest) continue;
            double dist = metrics::distance(data[static_cast<size_t>(i)].values,
                                            centroids[static_cast<size_t>(c)].values,
                                            metrics::GLOBAL_METRIC_CFG);
            if (dist < b_i) {
                b_i = dist;
            }
        }

        // Compute silhouette
        double sil = 0.0;
        double max_val = std::max(a_i, b_i);
        if (max_val > 0.0) sil = (b_i - a_i) / max_val;

        silhouettes[static_cast<size_t>(nearest)] += sil;
        assigned[static_cast<size_t>(nearest)]++;
        total_sil += sil;
    }

    // Compute final averages
    for (size_t i = 0; i < silhouettes.size(); ++i) {
        if (assigned[i] > 0) {
            silhouettes[i] /= static_cast<double>(assigned[i]);
        }
    }
    if (n_points_ > 0) total_sil /= static_cast<double>(n_points_);

    return {silhouettes, total_sil};
}

// Optimized silhouette computation with multiple speedup strategies
// Optimized silhouette computation
std::pair<std::vector<double>, double> IVFPQSearch::compute_silhouette() const {
    // Create cluster maps
    std::vector<std::vector<int>> centroids_map = get_centroids_map();

    // Results
    std::vector<double> silhouettes(static_cast<size_t>(p.kclusters), 0.0);
    std::vector<int> assigned(static_cast<size_t>(p.kclusters), 0);
    double total_sil = 0.0;

    for (int i = 0; i < n_points_; ++i) {
        int nearest = (i < static_cast<int>(data_assignments_.size())) ? data_assignments_[i] : -1;
        if (nearest < 0) continue;

        const auto& cluster_indices = centroids_map[static_cast<size_t>(nearest)];
        if (cluster_indices.empty()) continue;

        // Compute a_i: mean distance to same cluster
        double a_i = 0.0;
        if (cluster_indices.size() > 1) {
            std::vector<double> intra;
            intra.reserve(cluster_indices.size() - 1);
            for (int idx : cluster_indices) {
                if (idx != i) {
                    intra.push_back(metrics::distance(data[static_cast<size_t>(i)].values,
                                                       data[static_cast<size_t>(idx)].values,
                                                       metrics::GLOBAL_METRIC_CFG));
                }
            }
            if (!intra.empty()) a_i = our_math::mean(intra);
        }

        // Compute b_i: mean distance to second nearest cluster
        double b_i = std::numeric_limits<double>::max();
        int second = second_nearest_centroid(data[static_cast<size_t>(i)]);
        if (second >= 0) {
            const auto& other_indices = centroids_map[static_cast<size_t>(second)];
            if (!other_indices.empty()) {
                std::vector<double> inter;
                inter.reserve(other_indices.size());
                for (int idx : other_indices) {
                    inter.push_back(metrics::distance(data[static_cast<size_t>(i)].values,
                                                      data[static_cast<size_t>(idx)].values,
                                                      metrics::GLOBAL_METRIC_CFG));
                }
                if (!inter.empty()) b_i = our_math::mean(inter);
            }
        }

        // Compute silhouette
        double sil = 0.0;
        if (cluster_indices.size() > 1) {
            double max_val = std::max(a_i, b_i);
            if (max_val > 0.0 && b_i < std::numeric_limits<double>::max()) {
                sil = (b_i - a_i) / max_val;
            }
        }

        // Thread-safe accumulation
        silhouettes[static_cast<size_t>(nearest)] += sil;
        assigned[static_cast<size_t>(nearest)]++;
        total_sil += sil;
    }

    // Compute final averages
    for (size_t i = 0; i < silhouettes.size(); ++i) {
        if (assigned[i] > 0) {
            silhouettes[i] /= static_cast<double>(assigned[i]);
        }
    }
    if (n_points_ > 0) total_sil /= static_cast<double>(n_points_);

    return {silhouettes, total_sil};
}

// Coarse clustering helpers --------------------------------------------------

void IVFPQSearch::random_subset() {
    subset_data.clear();
    subset_assignments_.clear();

    if (n_points_ == 0) return;

    // 1. Initialize distribution
    std::uniform_int_distribution<size_t> distribution_int(0, static_cast<size_t>(n_points_ - 1));

    // 2. Keep track of already selected points
    std::vector<bool> selected(static_cast<size_t>(n_points_), false);

    // 3. Select random unique feature vectors
    size_t subset_size = static_cast<size_t>(std::sqrt(static_cast<double>(n_points_)));
    subset_size = std::max<size_t>(1, subset_size);
    subset_data.reserve(subset_size);

    for (size_t i = 0; i < subset_size; ++i) {
        size_t index;
        do {
            index = distribution_int(rng);
        } while (selected[index]); // ensure uniqueness

        selected[index] = true;

        // push chosen vector into subset
        subset_data.push_back(data[index]);
    }
    subset_assignments_.assign(subset_data.size(), 0);
}

void IVFPQSearch::initialization() {
    // Clear existing centroids before starting
    centroids.clear();
    if (subset_data.empty()) return;

    int subset_count = static_cast<int>(subset_data.size());

    std::uniform_int_distribution<int> dist(0, subset_count - 1);
    int index = dist(rng);
    // this is our first centroid
    centroids.push_back(subset_data[static_cast<size_t>(index)]);

    //4. until t==k go to 2.
    for (int t = 1; t < p.kclusters; ++t) {

        std::vector<double> D(subset_count, std::numeric_limits<double>::max()); // distance vector
        std::vector<double> P(subset_count, 0.0);                               // probability vector

        double sum_D_squared = 0.0;

        //2. every non-centroid find min distance from centroids
        // if it is centroid then the distance is zero  
        // and the probability of being chosen is zero
        for (int i = 0; i < subset_count; ++i) {

            // find min distance from all cetroids
            for (int j = 0; j < t; ++j) {
                double dist_ij = metrics::distance(
                    subset_data[static_cast<size_t>(i)].values,
                    centroids[static_cast<size_t>(j)].values,
                    metrics::GLOBAL_METRIC_CFG
                );

                if (dist_ij < D[static_cast<size_t>(i)])
                    D[static_cast<size_t>(i)] = dist_ij;
            }

            double d_squared = D[static_cast<size_t>(i)] * D[static_cast<size_t>(i)];
            sum_D_squared += d_squared;

            P[static_cast<size_t>(i)] = (i == 0) ? d_squared : P[static_cast<size_t>(i - 1)] + d_squared;

        }

        //  if all remaining points are already centroids (or sum_D_squared is 0)
        if (sum_D_squared == 0.0) break;

        // 3. Choose a new centroid based on D(x)^2 probability
        std::uniform_real_distribution<double> dist_real(0.0, P.back());
        double x = dist_real(rng); // x is the random value to search for
        int r_index = static_cast<int>(std::upper_bound(P.begin(), P.end(), x) - P.begin());

        // Sanity check to ensure we found a valid index
        if (r_index >= subset_count) r_index = subset_count - 1;

        centroids.push_back(subset_data[static_cast<size_t>(r_index)]);

    }
    subset_assignments_.assign(subset_data.size(), 0);
}

// assign vectors to the nearest cluster using the lloyds algorithm
int IVFPQSearch::assignment_lloyds() {
    // keep track of the changes in assignments
    int changes = 0;
    // itterate each vector of the dataset
    int sqrt_n_points = static_cast<int>(subset_data.size());

    for (int i = 0; i < sqrt_n_points; ++i) {
        int prev = subset_assignments_[static_cast<size_t>(i)];
        subset_assignments_[static_cast<size_t>(i)] = nearest_centroid(subset_data[static_cast<size_t>(i)]);
        // check and record changed centroids
        if (prev != subset_assignments_[static_cast<size_t>(i)])
            changes++;
    }

    return changes;
}

// update the centroids with the kmedian method
void IVFPQSearch::update() {
    if (centroids.empty()) return;

    // We want a map containing the true values of the vectors
    std::vector<std::vector<Vector>> centroids_map(static_cast<size_t>(p.kclusters));
    // insert the vectors in the map
    int sqrt_n_points = static_cast<int>(subset_data.size());

    for (int i = 0; i < sqrt_n_points; ++i) {
        int assigned = subset_assignments_[static_cast<size_t>(i)];
        if (assigned < 0 || assigned >= p.kclusters) continue;
        centroids_map[static_cast<size_t>(assigned)].push_back(subset_data[static_cast<size_t>(i)]);
    }
    // for each centroid 
    for (int i = 0; i < p.kclusters; ++i) {

        const auto& cluster_vectors = centroids_map[static_cast<size_t>(i)];

        // for each one of its dimenions
        for (int j = 0; j < space_dim_; ++j) {

            // Collect all components (values) for dimension j from all vectors in cluster i
            std::vector<double> current_component;
            current_component.reserve(cluster_vectors.size());
            
            for (const auto& assigned_vec : cluster_vectors) {
                // Access the specific dimension 'j' from the underlying values vector
                current_component.push_back(assigned_vec.values[static_cast<size_t>(j)]); 
            }
            
            // Find the median of the collected components
            if (!current_component.empty()) {
                double new_median = our_math::median(current_component);
                
                // Update the j-th component of the i-th centroid
                centroids[static_cast<size_t>(i)].values[static_cast<size_t>(j)] = new_median;
            }
        }
    }
}

int IVFPQSearch::nearest_centroid(const Vector& vec) const {
    double min_dist = std::numeric_limits<double>::max(); 
    int nearest = -1;
    
    // compute the distances to all the centroids
    for (int i = 0; i < static_cast<int>(centroids.size()); ++i) {
        double dist = metrics::distance(
            vec.values,
            centroids[static_cast<size_t>(i)].values,
            metrics::GLOBAL_METRIC_CFG
        );
        
        // set it as min
        if (dist < min_dist) {
            min_dist = dist;
            nearest = i;
        }
    }
    
    return nearest;
}

int IVFPQSearch::second_nearest_centroid(const Vector& vec) const {
    double min_dist = std::numeric_limits<double>::max(); 
    int nearest = nearest_centroid(vec);
    int second_nearest = -1;

    // compute the distances to all the centroids
    for (int i = 0; i < static_cast<int>(centroids.size()); ++i) {
        double dist = metrics::distance(
            vec.values,
            centroids[static_cast<size_t>(i)].values,
            metrics::GLOBAL_METRIC_CFG
        );
        
        // set it as min
        if (dist < min_dist && i != nearest) {
            min_dist = dist;
            second_nearest = i;
        }
    }
    
    return second_nearest;
}

// PQ helpers -----------------------------------------------------------------

// Step 3: compute residual r(x) = x - c(x)
std::vector<double> IVFPQSearch::compute_residual(const Vector& vec, int centroid_idx) const {
    std::vector<double> residual(space_dim_, 0.0);
    if (centroid_idx < 0 || centroid_idx >= static_cast<int>(centroids.size())) return residual;

    const auto& centroid = centroids[static_cast<size_t>(centroid_idx)].values;
    for (int d = 0; d < space_dim_; ++d) {
        residual[static_cast<size_t>(d)] = vec.values[static_cast<size_t>(d)] - centroid[static_cast<size_t>(d)];
    }
    return residual;
}

// Steps 4-5: split residuals into M subvectors and train the codebooks
void IVFPQSearch::build_pq_codebooks() {
    pq_codebooks_.assign(static_cast<size_t>(p.M), {});
    std::cout << "[IVFPQ] Building PQ codebooks...\n";
    std::vector<std::vector<Vector>> training(static_cast<size_t>(p.M));
    for (int idx = 0; idx < n_points_; ++idx) {
        int cid = data_assignments_[idx];
        if (cid < 0) continue;
        std::cout << "[IVFPQ] Processing vector " << idx << " for centroid " << cid << ".\n";
        std::vector<double> residual = compute_residual(data[static_cast<size_t>(idx)], cid);
        for (int m = 0; m < p.M; ++m) {
            Vector part;
            part.values.resize(static_cast<size_t>(subvector_dim_));
            size_t offset = static_cast<size_t>(m * subvector_dim_);
            for (int d = 0; d < subvector_dim_; ++d) {
                part.values[static_cast<size_t>(d)] = residual[offset + static_cast<size_t>(d)];
            }
            training[static_cast<size_t>(m)].push_back(std::move(part));
        }
        std::cout << "[IVFPQ] Vector " << idx << " processed.\n";
    }

    for (int m = 0; m < p.M; ++m) {
        std::cout << "[IVFPQ] Training PQ codebook for sub-vector " << m << ".\n";
        auto centers = kmeans_pp(training[static_cast<size_t>(m)],
                                 codebook_size_,
                                 subvector_dim_,
                                 rng);
        pq_codebooks_[static_cast<size_t>(m)] = std::move(centers);
        std::cout << "[IVFPQ] PQ codebook for sub-vector " << m << " trained with "
                  << pq_codebooks_[static_cast<size_t>(m)].size() << " centroids.\n";
    }
}

// Steps 6-7: encode PQ(x) = [code1,...,codeM] for the assigned centroid
std::vector<std::uint8_t> IVFPQSearch::encode_point(const Vector& vec, int centroid_idx) const {
    std::vector<std::uint8_t> codes(static_cast<size_t>(p.M), 0);
    if (centroid_idx < 0) return codes;
    std::vector<double> residual = compute_residual(vec, centroid_idx);
    for (int m = 0; m < p.M; ++m) {
        size_t offset = static_cast<size_t>(m * subvector_dim_);
        double best = std::numeric_limits<double>::max();
        std::uint8_t best_idx = 0;

        for (int h = 0; h < codebook_size_; ++h) {
            const auto& center = pq_codebooks_[static_cast<size_t>(m)][static_cast<size_t>(h)].values;
            double dist = 0.0;
            for (int d = 0; d < subvector_dim_; ++d) {
                double diff = residual[offset + static_cast<size_t>(d)] - center[static_cast<size_t>(d)];
                dist += diff * diff;
            }
            if (dist < best) {
                best = dist;
                best_idx = static_cast<std::uint8_t>(h);
            }
        }
        codes[static_cast<size_t>(m)] = best_idx;
    }
    return codes;
}