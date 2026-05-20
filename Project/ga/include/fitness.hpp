#pragma once

#include "graph.hpp"
#include "individual.hpp"

int evaluate_conflicts(const Graph& graph, const std::vector<int>& colors);
int evaluate_used_colors(const std::vector<int>& colors);
void evaluate_individual(const Graph& graph, Individual& individual);