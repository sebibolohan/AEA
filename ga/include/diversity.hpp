#pragma once

#include <vector>

#include "individual.hpp"

double normalized_hamming_distance(const Individual& a, const Individual& b);
double greedy_remapped_distance(const Individual& a, const Individual& b);
bool has_exact_same_coloring(const std::vector<Individual>& population, const Individual& candidate);