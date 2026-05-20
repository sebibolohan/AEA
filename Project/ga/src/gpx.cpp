#include "gpx.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "fitness.hpp"

namespace {
std::vector<std::vector<int>> build_color_classes(const Individual& individual, const int k) {
    std::vector<std::vector<int>> classes(static_cast<std::size_t>(k));
    for (int v = 0; v < static_cast<int>(individual.color.size()); ++v) {
        const int c = individual.color[static_cast<std::size_t>(v)] % k;
        classes[static_cast<std::size_t>(c)].push_back(v);
    }
    return classes;
}

int select_largest_remaining_class(
    const std::vector<std::vector<int>>& classes,
    const std::vector<unsigned char>& used_class,
    const std::vector<unsigned char>& assigned,
    std::mt19937_64& rng) {

    int best_size = -1;
    std::vector<int> ties;

    for (int c = 0; c < static_cast<int>(classes.size()); ++c) {
        if (used_class[static_cast<std::size_t>(c)] != 0U) {
            continue;
        }

        int current_size = 0;
        for (int v : classes[static_cast<std::size_t>(c)]) {
            if (assigned[static_cast<std::size_t>(v)] == 0U) {
                ++current_size;
            }
        }

        if (current_size > best_size) {
            best_size = current_size;
            ties.clear();
            ties.push_back(c);
        } else if (current_size == best_size) {
            ties.push_back(c);
        }
    }

    if (ties.empty()) {
        return -1;
    }

    std::uniform_int_distribution<std::size_t> dist(0U, ties.size() - 1U);
    return ties[dist(rng)];
}

int best_completion_color(
    const Graph& graph,
    const std::vector<int>& colors,
    const int v,
    const int k,
    std::mt19937_64& rng) {

    int best_conflicts = std::numeric_limits<int>::max();
    std::vector<int> ties;

    for (int c = 0; c < k; ++c) {
        int local_conflicts = 0;
        for (int u : graph.adj[static_cast<std::size_t>(v)]) {
            const int uc = colors[static_cast<std::size_t>(u)];
            if (uc == c) {
                ++local_conflicts;
            }
        }

        if (local_conflicts < best_conflicts) {
            best_conflicts = local_conflicts;
            ties.clear();
            ties.push_back(c);
        } else if (local_conflicts == best_conflicts) {
            ties.push_back(c);
        }
    }

    std::uniform_int_distribution<std::size_t> dist(0U, ties.size() - 1U);
    return ties[dist(rng)];
}
}  // namespace

Individual gpx_crossover(
    const Graph& graph,
    const int k,
    const Individual& parent1,
    const Individual& parent2,
    std::mt19937_64& rng) {

    if (k <= 0) {
        throw std::invalid_argument("GPX requires k > 0");
    }
    if (static_cast<int>(parent1.color.size()) != graph.n || static_cast<int>(parent2.color.size()) != graph.n) {
        throw std::invalid_argument("Parent size mismatch in GPX");
    }

    const auto classes1 = build_color_classes(parent1, k);
    const auto classes2 = build_color_classes(parent2, k);

    std::vector<unsigned char> used1(static_cast<std::size_t>(k), 0U);
    std::vector<unsigned char> used2(static_cast<std::size_t>(k), 0U);
    std::vector<unsigned char> assigned(static_cast<std::size_t>(graph.n), 0U);
    std::vector<int> child_colors(static_cast<std::size_t>(graph.n), -1);

    for (int slot = 0; slot < k; ++slot) {
        const bool take_from_first = (slot % 2 == 0);
        const auto& classes = take_from_first ? classes1 : classes2;
        auto& used = take_from_first ? used1 : used2;

        const int selected = select_largest_remaining_class(classes, used, assigned, rng);
        if (selected == -1) {
            continue;
        }

        used[static_cast<std::size_t>(selected)] = 1U;
        for (int v : classes[static_cast<std::size_t>(selected)]) {
            if (assigned[static_cast<std::size_t>(v)] == 0U) {
                assigned[static_cast<std::size_t>(v)] = 1U;
                child_colors[static_cast<std::size_t>(v)] = slot;
            }
        }
    }

    for (int v = 0; v < graph.n; ++v) {
        if (child_colors[static_cast<std::size_t>(v)] == -1) {
            child_colors[static_cast<std::size_t>(v)] =
                best_completion_color(graph, child_colors, v, k, rng);
        }
    }

    Individual child;
    child.color = std::move(child_colors);
    evaluate_individual(graph, child);
    return child;
}