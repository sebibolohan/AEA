#include "fitness.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

int evaluate_conflicts(const Graph& graph, const std::vector<int>& colors) {
    if (static_cast<int>(colors.size()) != graph.n) {
        throw std::invalid_argument("Color vector size does not match graph.n");
    }

    int conflicts = 0;
    for (const auto& [u, v] : graph.edges) {
        if (colors[static_cast<std::size_t>(u)] == colors[static_cast<std::size_t>(v)]) {
            ++conflicts;
        }
    }
    return conflicts;
}

int evaluate_used_colors(const std::vector<int>& colors) {
    int max_color = -1;
    for (int c : colors) {
        if (c < 0) {
            throw std::invalid_argument("Color vector contains negative color");
        }
        max_color = std::max(max_color, c);
    }
    return max_color + 1;
}

void evaluate_individual(const Graph& graph, Individual& individual) {
    individual.conflict_count = evaluate_conflicts(graph, individual.color);
    individual.used_colors = evaluate_used_colors(individual.color);
}