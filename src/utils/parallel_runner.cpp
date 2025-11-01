#include <thread>
#include <atomic>
#include <iostream>

#include "../../include/utils/parallel_runner.h"

std::vector<SearchResult> run_parallel_search(
    const SearchAlgorithm* algo,
    const std::vector<Vector>& queries,
    int num_threads,
    const Params& params
) {
    std::vector<SearchResult> results(queries.size());
    std::atomic<int> counter(0);

    auto worker = [&]() {
        while (true) {
            int i = counter++;
            if (i >= (int)queries.size()) break;
            results[i] = algo->search(queries[i], params, i);
            // if(i == 10 ){
            //     std::cout << "sanity check parallel_runner \n \t quiry counter: " << i << std::endl;
            //     break;
            // }
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t){

        threads.emplace_back(worker);
    }

    for (auto& t : threads) {

        t.join();
    }

    std::cout << "[Parallel] Completed all queries with " << num_threads << " threads.\n";
    return results;
}