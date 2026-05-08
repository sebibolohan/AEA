#pragma once

#include <random>

#include "graph.hpp"
#include "individual.hpp"

Individual gpx_crossover(
    const Graph& graph,
    int k,
    const Individual& parent1,
    const Individual& parent2,
    std::mt19937_64& rng);