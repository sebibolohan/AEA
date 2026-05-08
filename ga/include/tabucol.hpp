#pragma once

#include <cstdint>
#include <random>

#include "config.hpp"
#include "graph.hpp"
#include "individual.hpp"

struct TabuColResult {
    bool solved = false;
    int best_conflicts = 0;
    std::uint64_t iterations_used = 0;
};

TabuColResult run_tabucol(
    const Graph& graph,
    int k,
    Individual& individual,
    const SolverConfig& config,
    std::mt19937_64& rng);