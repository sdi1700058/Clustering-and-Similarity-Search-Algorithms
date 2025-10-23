#ifndef DUMMY_SEARCH_H
#define DUMMY_SEARCH_H

/*
!PLACEHOLDER IMPLEMENTATION OF A DUMMY SEARCH ALGORITHM!
This serves basically as a test to check if the main program
and the overall structure is working fine (builds and runs successfully).
*/

#include <iostream>

#include "search_algorithm.h"

class DummySearch : public SearchAlgorithm {
private:
    std::vector<Vector> data;
public:
    void build_index(const std::vector<Vector>& dataset) override;
    SearchResult search(const Vector& query, const Params& params, int query_id) const override;
};

#endif