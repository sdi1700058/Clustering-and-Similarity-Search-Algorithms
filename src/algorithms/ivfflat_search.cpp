#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <cassert>

#include "../../include/algorithms/ivfflat_search.h"
#include "../../include/utils/args_parser.h"
#include "../../include/common/metrics.h"
#include "../../include/common/our_math.h"


void IVFFlatSearch::configure(const Args& args) {
    p.seed = args.seed;
    p.kclusters = args.kclusters;
    p.nprobe = args.nprobe;
    p.N = args.N;
    p.R = args.R;
    rng.seed(p.seed);
}

void IVFFlatSearch::build_index(const std::vector<Vector>& dataset) {
    data = dataset;
    space_dim_ = dataset.empty() ? 0 : static_cast<int>(dataset.front().values.size());
    n_points = dataset.empty() ? 0 : static_cast<int>(dataset.size());
    centroids.clear();

    n_points = data.size();
    assert(n_points);
    std::vector<int> assigned(n_points, 0);
    assigned_centroid = assigned;

    // 1.Lloyd's Clustering
    random_subset();

    // Step 1.a: Initialize the centroids using init++
    initialization();

    // set manually a threshold for stoping the algorithm
    int threshold = 1;
    // number of changed vector. It is going to change in each itteration
    int changed = std::numeric_limits<int>::max(); ;

    // repeat until the threshold is not reached
    while (changed > threshold) {
        // Step 1.b: call the appropriate method of assignment
        changed = assignment_lloyds();
    
        // Step 1.c: Update the centroids
        update();
    }
    assigned_centroid.clear();
    IL.clear();
    IL.resize(p.kclusters);
    // 2. Build Inverted Lists
    for (int i = 0; i < n_points; i++) {
        // 2.Assign to nearest centroid
        assigned_centroid[i] = nearest_centroid(data[i]);
        // 3.Append to Inverted List
        IL[assigned_centroid[i]].push_back({i, data[i]});
    }

    index_built = true;
    std::cout << "[IVFFlat] index built with " << data.size() << " vectors, k=" << p.kclusters << "\n";
}

SearchResult IVFFlatSearch::search(const Vector& query, const Params& params, int query_id) const {
    auto t0 = std::chrono::high_resolution_clock::now();
    SearchResult res; 
    res.query_id = query_id;
    
    if (data.empty() || space_dim_ == 0 || static_cast<int>(query.values.size()) != space_dim_) {
        res.time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        return res;
    }

    std::vector<std::pair<int, double>> S; // centroid_index, dist
    S.reserve((int)p.kclusters); 

    // 1. Distance to all centroids & select top 'nprobes'
    
    // a. Calculate distance to all k centroids
    for (int j = 0; j < (int)p.kclusters; ++j) {
        double dist = metrics::distance(
            query.values, 
            centroids[j].values, 
            metrics::GLOBAL_METRIC_CFG
        );
        S.push_back({j, dist}); 
    }

    // b. Select the top 'nprobes' closest centroids
    size_t effective_nprobes = std::min((size_t)p.nprobe, S.size());
    
    // Use std::nth_element to quickly find the top effective_nprobes
    std::nth_element(
        S.begin(), 
        S.begin() + effective_nprobes, 
        S.end(), 
        [](const auto& a, const auto& b) { return a.second < b.second; }
    );
    S.resize(effective_nprobes);

    // 2. Compute U (b)
    std::vector<std::pair<int, double>> b; // candidate_index, dist
    
    // Iterate through the selected 'nprobes' centroids in S
    for (const auto& g : S) { // g is {centroid_index, dist_to_q}

        // Iterate through all data points assigned to this centroid (IL[centroid_idx])
        for (const auto& entry : IL[g.first]) { // entry is {data_index, data[data_index]}

            // Calculate the distance from query 'q' to the data point 'x'
            double dist = metrics::distance(
                    query.values, 
                    entry.second.values, // Access full dataset using index
                    metrics::GLOBAL_METRIC_CFG
                );
            b.push_back({entry.first, dist});
        }
    }

    // 3. Find the R nearest b

    if (!b.empty()) {
        
        std::sort( b.begin(), b.end(), [](const auto& a, const auto& b){ return a.second < b.second; } );
        
        int topK = std::min(params.N, static_cast<int>(b.size()));
        topK = std::max(topK, 0);
        for (int i = 0; i < topK; ++i) {
            res.neighbor_ids.push_back(b[i].first);
            res.distances.push_back(static_cast<float>(b[i].second));
        }
        
        if (params.enable_range && params.R > 0.0) {
            for (const auto& cand : b) {
                if (cand.second <= params.R) {
                    res.range_neighbor_ids.push_back(cand.first);
                    res.range_distances.push_back(static_cast<float>(cand.second));
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
        
}

int IVFFlatSearch::nearest_centroid(const Vector& vec) {
    double min_dist = std::numeric_limits<double>::max(); 
    int nearest = -1;
    
    // compute the distances to all the centroids
    for (int i = 0; i < (int)p.kclusters; i++) {
        double dist = metrics::distance(
            vec.values,                  // Input vector's values
            centroids[i].values,      // Centroid i's values
            metrics::GLOBAL_METRIC_CFG
        );
        
        // set it as min
        if (dist < min_dist) {
            min_dist = dist;
            nearest = i;
        }
    }
    
    assert(nearest != -1); 

    return nearest;
}

int IVFFlatSearch::second_nearest_centroid(const Vector& vec) {
    double min_dist = std::numeric_limits<double>::max(); 
    int nearest = nearest_centroid(vec);
    int second_nearest = -1;

    // compute the distances to all the centroids
    for (int i = 0; i < (int)p.kclusters; i++) {
        double dist = metrics::distance(
            vec.values,                  // Input vector's values
            centroids[i].values,      // Centroid i's values
            metrics::GLOBAL_METRIC_CFG
        );
        
        // set it as min
        if (dist < min_dist && i != nearest) {
            min_dist = dist;
            second_nearest = i;
        }
    }
    
    assert(second_nearest != -1); 

    return second_nearest;
}

// get the vector of the centroids
std::vector<Vector> IVFFlatSearch::get_centroids() {
    return this->centroids;
};

std::vector<std::vector<int>> IVFFlatSearch::get_centroids_map() {
    // we want to keep a map containing the true values of the vectors, in order to find the median
    std::vector<std::vector<int>> centroids_map((int)p.kclusters);
    // insert the vectors in the map
    for (int i = 0; i < n_points; i++) {
        int assigned = assigned_centroid[i];
        centroids_map[assigned].push_back(i);
    }	
    return centroids_map;
}

void IVFFlatSearch::random_subset() {

    // 1. Initialize distribution
    std::uniform_int_distribution<size_t> distribution_int(0, n_points - 1);

    // 2. Keep track of already selected points
    std::vector<bool> selected(n_points, false);

    // 3. Select random unique feature vectors
    size_t subset_size = static_cast<size_t>(std::sqrt(n_points));
    for (size_t i = 0; i < subset_size; ++i) {
        size_t index;
        do {
            index = distribution_int(rng);
        } while (selected[index]); // ensure uniqueness

        selected[index] = true;

        // push chosen vector into subset
        subset_data.push_back(data[index]);
    }

}

// initialize our clusters using the init++ method
void IVFFlatSearch::initialization() {

    // Clear existing centroids before starting
    centroids.clear();
    
    int sqrt_n_points = static_cast<int>(subset_data.size());

    std::uniform_int_distribution<int> distribution_int(0,sqrt_n_points - 1);
    int index = distribution_int(rng);
    // this is our first centroid
    centroids.push_back(subset_data[index]);

    //4. until t==k go to 2.
    for ( int t = 1; t < (int)p.kclusters; t++){

        std::vector<double> D(sqrt_n_points, std::numeric_limits<double>::max()); // distance vector
        std::vector<double> P(sqrt_n_points, 0.0);                               // probability vector

        double sum_D_squared = 0.0;

        //2. every non-centroid find min distance from centroids
        // if it is centroid then the distance is zero  
        // and the probability of being chosen is zero
        for (int i = 0; i < sqrt_n_points; i++) {

            // find min distance from all cetroids
            for (int j = 0; j < t; j++) {
                
                double dist = metrics::distance(
                    subset_data[i].values, 
                    centroids[j].values, 
                    metrics::GLOBAL_METRIC_CFG // Assuming a global metric is defined
                );

                if (dist < D[i])
                    D[i] = dist;
            }

            double d_squared = D[i] * D[i]; 
            sum_D_squared += d_squared;
            
            P[i] = (i == 0) ? d_squared : P[i - 1] + d_squared;
            
        }

        //  if all remaining points are already centroids (or sum_D_squared is 0)
        if (sum_D_squared == 0.0) break; 
        
        // 3. Choose a new centroid based on D(x)^2 probability
        std::uniform_real_distribution<double> distribution_real(0.0, P.back()); 
        double x = distribution_real(rng); // x is the random value to search for
        int r_index = std::upper_bound(P.begin(), P.end(), x) - P.begin();

        // Sanity check to ensure we found a valid index
        assert(r_index >= 0 && r_index < sqrt_n_points); 

        centroids.push_back(subset_data[r_index]);
        
        
    }
    std::cout << "sanity check: initialization \n \t" << "centroids: " << centroids.size() << std::endl;

}

// update the centroids with the kmedian method
void IVFFlatSearch::update() {
    // We want a map containing the true values of the vectors
    std::vector<std::vector<Vector>> centroids_map((int)p.kclusters);
    // insert the vectors in the map
    int sqrt_n_points = static_cast<int>(subset_data.size());

    for (int i = 0; i < sqrt_n_points; i++) {
        int assigned = assigned_centroid[i];
        centroids_map[assigned].push_back(subset_data[i]);
    }
    // for each centroid 
    for (int i = 0; i < (int)p.kclusters; i++) {

        const auto& cluster_vectors = centroids_map[i];

        // for each one of its dimenions
        for (int j = 0; j < (int)space_dim_; j++) {

            // Collect all components (values) for dimension j from all vectors in cluster i
            std::vector<double> current_component;
            current_component.reserve(cluster_vectors.size());
            
            for (const auto& assigned_vec : cluster_vectors) {
                // Access the specific dimension 'j' from the underlying values vector
                current_component.push_back(assigned_vec.values[j]); 
            }
            
            // Find the median of the collected components
            float new_median = our_math::median(current_component);
            
            // Update the j-th component of the i-th centroid
            centroids[i].values[j] = new_median;
        }
    }
};

// assign vectors to the nearest cluster using the lloyds algorithm
int IVFFlatSearch::assignment_lloyds() {
    // keep track of the changes in assignments
    int changes = 0;
    // itterate each vector of the dataset
    int sqrt_n_points = static_cast<int>(subset_data.size());

    for (int i = 0; i < sqrt_n_points; i++) {
        int prev = assigned_centroid[i];
        assigned_centroid[i] = nearest_centroid(subset_data[i]);
        // check and record changed centroids
        if (prev != assigned_centroid[i])
            changes++;
    }

    return changes;
};


// FAST APPROXIMATION VERSION: Use centroids instead of all points
// This is 10-100x faster with minimal accuracy loss
std::pair<std::vector<double>, double> IVFFlatSearch::compute_silhouette_fast() {
    
    std::vector<double> silhouettes((int)p.kclusters, 0);
    std::vector<int> assigned((int)p.kclusters, 0);
    double total_sil = 0;
    

    for (int i = 0; i < n_points; i++) {
        int nearest = assigned_centroid[i];
        
        // a_i: distance to own centroid
        double a_i = metrics::distance(
            data[i].values, 
            centroids[nearest].values, 
            metrics::GLOBAL_METRIC_CFG
        );
        
        // b_i: distance to nearest other centroid
        double b_i = std::numeric_limits<double>::max();
        for (int c = 0; c < (int)p.kclusters; c++) {
            if (c != nearest) {
                double dist = metrics::distance(
                    data[i].values, 
                    centroids[c].values, 
                    metrics::GLOBAL_METRIC_CFG
                );

                if (dist < b_i) {
                    b_i = dist;
                }
            }
        }
        
        // Compute silhouette
        double sil = 0;
        double max_val = std::max(a_i, b_i);

        if (max_val > 0) sil = (b_i - a_i) / max_val;
        
        silhouettes[nearest] += sil;
        assigned[nearest]++;
        total_sil += sil;
    }
    
    // Compute final averages
    for (int i = 0; i < (int)p.kclusters; i++) {
        if (assigned[i] > 0) {
            silhouettes[i] /= assigned[i];
        }
    }
    
    total_sil /= n_points;
    
    return std::make_pair(silhouettes, total_sil);
}

// Optimized silhouette computation with multiple speedup strategies
// Optimized silhouette computation
std::pair<std::vector<double>, double> IVFFlatSearch::compute_silhouette() {
    
    // Create cluster maps
    std::vector<std::vector<int>> centroids_map((int)p.kclusters);
    
    for (int i = 0; i < (int)p.kclusters; i++) {
        centroids_map[i].reserve(n_points / (int)p.kclusters + 100);
    }
    
    for (int i = 0; i < n_points; i++) {
        centroids_map[assigned_centroid[i]].push_back(i);
    }
    
    // Results
    std::vector<double> silhouettes((int)p.kclusters, 0);
    std::vector<int> assigned((int)p.kclusters, 0);
    double total_sil = 0;
    
    for (int i = 0; i < n_points; i++) {

        int nearest = assigned_centroid[i];
        const auto& cluster = centroids_map[nearest];
        size_t cluster_size = cluster.size();
        
        // Compute a_i: mean distance to same cluster
        double a_i = 0;
        
        if (cluster_size > 1) {
            std::vector<double> a_i_dists;
            a_i_dists.reserve(cluster_size - 1);

            for (const int idx : cluster) {
                if (idx != i) {
                    a_i_dists.push_back(metrics::distance(
                        data[i].values, 
                        data[idx].values, 
                        metrics::GLOBAL_METRIC_CFG
                    ));
                }
            }
            a_i = our_math::mean(a_i_dists);
        }
        
        // Compute b_i: mean distance to second nearest cluster
        double b_i = 0;
        int second_nearest = second_nearest_centroid(data[i]);
        const auto& second_cluster_indices = centroids_map.at((size_t)second_nearest);
        std::vector<double> b_i_dists;
        b_i_dists.reserve(second_cluster_indices.size());
            
        for (const int idx : centroids_map[second_nearest]) {
            b_i_dists.push_back(metrics::distance(
                data[i].values, 
                data[idx].values, 
                metrics::GLOBAL_METRIC_CFG
            ));
        }

        if (!b_i_dists.empty()) b_i = our_math::mean(b_i_dists);
        
        // Compute silhouette
        double sil = 0;
        if (cluster_size > 1) {
            double max_val = std::max(a_i, b_i);
            if (max_val > 0) {
                sil = (b_i - a_i) / max_val;
            }
        }
        
        // Thread-safe accumulation
        silhouettes[nearest] += sil;
        assigned[nearest]++;
        total_sil += sil;
    }
    
    // Compute final averages
    for (int i = 0; i < (int)p.kclusters; i++) {
        if (assigned[i] > 0) {
            silhouettes[i] /= assigned[i];
        }
    }
    
    total_sil /= n_points;
    
    return std::make_pair(silhouettes, total_sil);
}


