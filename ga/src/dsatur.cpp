#include "dsatur.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fitness.hpp"

namespace {
int pick_vertex_for_dsatur(
    const Graph& graph,
    const std::vector<int>& color,
    const std::vector<int>& saturation_degree,
    std::mt19937_64& rng) {

    int best_sat = -1;
    int best_deg = -1;
    std::vector<int> ties;

    for (int v = 0; v < graph.n; ++v) {
        if (color[static_cast<std::size_t>(v)] != -1) {
            continue;
        }

        const int sat = saturation_degree[static_cast<std::size_t>(v)];
        const int deg = graph.degree[static_cast<std::size_t>(v)];

        if (sat > best_sat || (sat == best_sat && deg > best_deg)) {
            best_sat = sat;
            best_deg = deg;
            ties.clear();
            ties.push_back(v);
        } else if (sat == best_sat && deg == best_deg) {
            ties.push_back(v);
        }
    }

    if (ties.empty()) {
        throw std::runtime_error("DSATUR failed to select an uncolored vertex");
    }

    if (ties.size() == 1U) {
        return ties.front();
    }

    std::uniform_int_distribution<std::size_t> dist(0U, ties.size() - 1U);
    return ties[dist(rng)];
}
}  // namespace

DsaturSeedResult build_dsatur_seed(const Graph& graph, std::mt19937_64& rng) {
    if (graph.empty()) {
        return {};
    }

    const int n = graph.n;
    std::vector<int> color(static_cast<std::size_t>(n), -1);
    std::vector<int> saturation_degree(static_cast<std::size_t>(n), 0);
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(n));

    std::vector<std::vector<unsigned char>> neighbor_color_seen(
        static_cast<std::size_t>(n),
        std::vector<unsigned char>(static_cast<std::size_t>(n), 0U));

    int used_colors = 0;

    for (int step = 0; step < n; ++step) {
        const int v = pick_vertex_for_dsatur(graph, color, saturation_degree, rng);
        order.push_back(v);

        int chosen_color = 0;
        while (chosen_color < used_colors &&
               neighbor_color_seen[static_cast<std::size_t>(v)][static_cast<std::size_t>(chosen_color)] != 0U) {
            ++chosen_color;
        }

        color[static_cast<std::size_t>(v)] = chosen_color;
        if (chosen_color == used_colors) {
            ++used_colors;
        }

        for (int u : graph.adj[static_cast<std::size_t>(v)]) {
            if (color[static_cast<std::size_t>(u)] != -1) {
                continue;
            }

            auto& seen = neighbor_color_seen[static_cast<std::size_t>(u)][static_cast<std::size_t>(chosen_color)];
            if (seen == 0U) {
                seen = 1U;
                ++saturation_degree[static_cast<std::size_t>(u)];
            }
        }
    }

    Individual seed;
    seed.color = std::move(color);
    evaluate_individual(graph, seed);

    return {std::move(order), std::move(seed)};
}

std::vector<int> perturb_order(
    const std::vector<int>& base_order,
    const int swaps,
    std::mt19937_64& rng) {

    std::vector<int> order = base_order;
    if (order.size() < 2U || swaps <= 0) {
        return order;
    }

    std::uniform_int_distribution<std::size_t> dist(0U, order.size() - 1U);
    for (int i = 0; i < swaps; ++i) {
        const std::size_t a = dist(rng);
        const std::size_t b = dist(rng);
        std::swap(order[a], order[b]);
    }

    return order;
}

Individual construct_k_coloring_from_order(
    const Graph& graph,
    const std::vector<int>& order,
    const int k,
    std::mt19937_64& rng) {

    if (k <= 0) {
        throw std::invalid_argument("construct_k_coloring_from_order requires k > 0");
    }
    if (static_cast<int>(order.size()) != graph.n) {
        throw std::invalid_argument("Order size mismatch");
    }

    std::vector<int> color(static_cast<std::size_t>(graph.n), -1);

    for (int v : order) {
        std::vector<int> best_colors;
        int best_conflicts = std::numeric_limits<int>::max();

        for (int c = 0; c < k; ++c) {
            int local_conflicts = 0;
            for (int u : graph.adj[static_cast<std::size_t>(v)]) {
                const int uc = color[static_cast<std::size_t>(u)];
                if (uc == c) {
                    ++local_conflicts;
                }
            }

            if (local_conflicts < best_conflicts) {
                best_conflicts = local_conflicts;
                best_colors.clear();
                best_colors.push_back(c);
            } else if (local_conflicts == best_conflicts) {
                best_colors.push_back(c);
            }
        }

        if (best_colors.empty()) {
            throw std::runtime_error("No admissible color found while building k-coloring");
        }

        std::uniform_int_distribution<std::size_t> dist(0U, best_colors.size() - 1U);
        color[static_cast<std::size_t>(v)] = best_colors[dist(rng)];
    }

    Individual individual;
    individual.color = std::move(color);
    evaluate_individual(graph, individual);
    return individual;
}