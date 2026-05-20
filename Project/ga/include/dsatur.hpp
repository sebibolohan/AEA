#pragma once

#include <random>
#include <vector>

#include "graph.hpp"
#include "individual.hpp"

struct DsaturSeedResult {
    std::vector<int> order;
    Individual full_coloring;
};

DsaturSeedResult build_dsatur_seed(const Graph& graph, std::mt19937_64& rng);

std::vector<int> perturb_order(
    const std::vector<int>& base_order,
    int swaps,
    std::mt19937_64& rng);

Individual construct_k_coloring_from_order(
    const Graph& graph,
    const std::vector<int>& order,
    int k,
    std::mt19937_64& rng);