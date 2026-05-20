#include "tabucol.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fitness.hpp"

namespace {
int random_tabu_tenure(const SolverConfig& config, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> dist(config.tabu_tenure_min, config.tabu_tenure_max);
    return dist(rng);
}
}  // namespace

TabuColResult run_tabucol(
    const Graph& graph,
    const int k,
    Individual& individual,
    const SolverConfig& config,
    std::mt19937_64& rng) {

    if (k <= 0) {
        throw std::invalid_argument("TabuCol requires k > 0");
    }
    if (static_cast<int>(individual.color.size()) != graph.n) {
        throw std::invalid_argument("Individual size does not match graph.n");
    }

    for (int& c : individual.color) {
        if (c < 0) {
            c = 0;
        }
        c %= k;
    }

    const int n = graph.n;
    std::vector<std::vector<int>> neighbor_color_count(
        static_cast<std::size_t>(n),
        std::vector<int>(static_cast<std::size_t>(k), 0));

    for (int v = 0; v < n; ++v) {
        for (int u : graph.adj[static_cast<std::size_t>(v)]) {
            const int c = individual.color[static_cast<std::size_t>(u)];
            ++neighbor_color_count[static_cast<std::size_t>(v)][static_cast<std::size_t>(c)];
        }
    }

    std::vector<int> vertex_conflicts(static_cast<std::size_t>(n), 0);
    int current_conflicts = 0;
    for (int v = 0; v < n; ++v) {
        const int c = individual.color[static_cast<std::size_t>(v)];
        vertex_conflicts[static_cast<std::size_t>(v)] =
            neighbor_color_count[static_cast<std::size_t>(v)][static_cast<std::size_t>(c)];
        current_conflicts += vertex_conflicts[static_cast<std::size_t>(v)];
    }
    current_conflicts /= 2;

    std::vector<std::vector<int>> tabu_until(
        static_cast<std::size_t>(n),
        std::vector<int>(static_cast<std::size_t>(k), 0));

    std::vector<int> best_colors = individual.color;
    int best_conflicts = current_conflicts;

    for (int iteration = 1; iteration <= config.max_tabucol_iterations; ++iteration) {
        if (current_conflicts == 0) {
            individual.color = best_colors = individual.color;
            individual.conflict_count = 0;
            individual.used_colors = k;
            return {true, 0, static_cast<std::uint64_t>(iteration - 1)};
        }

        std::vector<int> conflicting_vertices;
        conflicting_vertices.reserve(static_cast<std::size_t>(n));
        for (int v = 0; v < n; ++v) {
            if (vertex_conflicts[static_cast<std::size_t>(v)] > 0) {
                conflicting_vertices.push_back(v);
            }
        }

        if (conflicting_vertices.empty()) {
            break;
        }

        int best_v = -1;
        int best_new_color = -1;
        int best_delta = std::numeric_limits<int>::max();
        std::vector<std::pair<int, int>> tied_moves;

        for (int v : conflicting_vertices) {
            const int old_color = individual.color[static_cast<std::size_t>(v)];
            const int removed = neighbor_color_count[static_cast<std::size_t>(v)][static_cast<std::size_t>(old_color)];

            for (int new_color = 0; new_color < k; ++new_color) {
                if (new_color == old_color) {
                    continue;
                }

                const int added =
                    neighbor_color_count[static_cast<std::size_t>(v)][static_cast<std::size_t>(new_color)];
                const int delta = added - removed;
                const int new_conflicts = current_conflicts + delta;
                const bool tabu =
                    tabu_until[static_cast<std::size_t>(v)][static_cast<std::size_t>(new_color)] > iteration;
                const bool aspiration = new_conflicts < best_conflicts;

                if (tabu && !aspiration) {
                    continue;
                }

                if (delta < best_delta) {
                    best_delta = delta;
                    best_v = v;
                    best_new_color = new_color;
                    tied_moves.clear();
                    tied_moves.emplace_back(v, new_color);
                } else if (delta == best_delta) {
                    tied_moves.emplace_back(v, new_color);
                }
            }
        }

        if (tied_moves.empty()) {
            break;
        }

        {
            std::uniform_int_distribution<std::size_t> dist(0U, tied_moves.size() - 1U);
            const auto [chosen_v, chosen_color] = tied_moves[dist(rng)];
            best_v = chosen_v;
            best_new_color = chosen_color;
        }

        const int old_color = individual.color[static_cast<std::size_t>(best_v)];
        const int removed = neighbor_color_count[static_cast<std::size_t>(best_v)][static_cast<std::size_t>(old_color)];
        const int added =
            neighbor_color_count[static_cast<std::size_t>(best_v)][static_cast<std::size_t>(best_new_color)];
        const int delta = added - removed;

        individual.color[static_cast<std::size_t>(best_v)] = best_new_color;
        current_conflicts += delta;

        for (int u : graph.adj[static_cast<std::size_t>(best_v)]) {
            --neighbor_color_count[static_cast<std::size_t>(u)][static_cast<std::size_t>(old_color)];
            ++neighbor_color_count[static_cast<std::size_t>(u)][static_cast<std::size_t>(best_new_color)];

            if (individual.color[static_cast<std::size_t>(u)] == old_color) {
                --vertex_conflicts[static_cast<std::size_t>(u)];
            }
            if (individual.color[static_cast<std::size_t>(u)] == best_new_color) {
                ++vertex_conflicts[static_cast<std::size_t>(u)];
            }
        }

        vertex_conflicts[static_cast<std::size_t>(best_v)] =
            neighbor_color_count[static_cast<std::size_t>(best_v)][static_cast<std::size_t>(best_new_color)];

        tabu_until[static_cast<std::size_t>(best_v)][static_cast<std::size_t>(old_color)] =
            iteration + random_tabu_tenure(config, rng);

        if (current_conflicts < best_conflicts) {
            best_conflicts = current_conflicts;
            best_colors = individual.color;
            if (best_conflicts == 0) {
                individual.color = best_colors;
                individual.conflict_count = 0;
                individual.used_colors = k;
                return {true, 0, static_cast<std::uint64_t>(iteration)};
            }
        }
    }

    individual.color = std::move(best_colors);
    individual.conflict_count = best_conflicts;
    individual.used_colors = k;
    return {best_conflicts == 0, best_conflicts, static_cast<std::uint64_t>(config.max_tabucol_iterations)};
}