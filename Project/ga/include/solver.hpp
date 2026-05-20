#pragma once

#include "config.hpp"
#include "graph.hpp"
#include "results.hpp"

SolveResult solve_graph_coloring(
    const Graph& graph,
    const std::string& instance_path,
    const SolverConfig& config);