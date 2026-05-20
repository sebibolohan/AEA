#include "solver.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef GA_USE_OPENMP
#include <omp.h>
#endif

#include "dsatur.hpp"
#include "fitness.hpp"
#include "gpx.hpp"
#include "population.hpp"
#include "random.hpp"
#include "tabucol.hpp"
#include "timer.hpp"

namespace {
struct FixedKRunResult {
    bool solved = false;
    int best_conflicts = -1;
    int generations = 0;
    std::uint64_t tabu_iterations = 0;
    Individual best_individual;
};

[[nodiscard]] std::uint64_t mix_seed(const std::uint64_t base, const std::uint64_t salt) {
    std::uint64_t x = base + 0x9e3779b97f4a7c15ULL + (salt << 6U) + (salt >> 2U);
    x ^= x >> 30U;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27U;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31U;
    return x;
}

Individual build_initial_candidate(
    const Graph& graph,
    const std::vector<int>& base_order,
    const int k,
    const SolverConfig& config,
    const std::uint64_t seed) {

    std::mt19937_64 rng(seed);
    const std::vector<int> perturbed = perturb_order(base_order, config.dsatur_order_perturb_swaps, rng);
    return construct_k_coloring_from_order(graph, perturbed, k, rng);
}

std::vector<Individual> initialize_population(
    const Graph& graph,
    const std::vector<int>& base_order,
    const int k,
    const SolverConfig& config,
    const std::uint64_t base_seed,
    std::uint64_t& total_tabu_iterations) {

    std::vector<Individual> population(static_cast<std::size_t>(config.population_size));

#ifdef GA_USE_OPENMP
    if (config.use_openmp) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < config.population_size; ++i) {
            const std::uint64_t local_seed = mix_seed(base_seed, static_cast<std::uint64_t>(i + 1));
            std::mt19937_64 rng(local_seed);

            Individual individual = build_initial_candidate(graph, base_order, k, config, local_seed);
            const TabuColResult tabu = run_tabucol(graph, k, individual, config, rng);

#pragma omp critical
            {
                total_tabu_iterations += tabu.iterations_used;
            }

            population[static_cast<std::size_t>(i)] = std::move(individual);
        }
    } else
#endif
    {
        for (int i = 0; i < config.population_size; ++i) {
            const std::uint64_t local_seed = mix_seed(base_seed, static_cast<std::uint64_t>(i + 1));
            std::mt19937_64 rng(local_seed);

            Individual individual = build_initial_candidate(graph, base_order, k, config, local_seed);
            const TabuColResult tabu = run_tabucol(graph, k, individual, config, rng);
            total_tabu_iterations += tabu.iterations_used;

            population[static_cast<std::size_t>(i)] = std::move(individual);
        }
    }

    return population;
}

FixedKRunResult run_fixed_k(
    const Graph& graph,
    const int k,
    const SolverConfig& config,
    std::mt19937_64& rng,
    const std::vector<int>& base_order,
    const std::uint64_t base_seed) {

    if (k <= 0) {
        throw std::invalid_argument("run_fixed_k requires k > 0");
    }

    std::uint64_t total_tabu_iterations = 0;
    int total_generations = 0;
    int restart_count = 0;

    std::vector<Individual> initial_population =
        initialize_population(graph, base_order, k, config, mix_seed(base_seed, static_cast<std::uint64_t>(k)), total_tabu_iterations);

    for (const auto& individual : initial_population) {
        if (individual.conflict_count == 0) {
            return {true, 0, 0, total_tabu_iterations, individual};
        }
    }

    Population population(std::move(initial_population));
    Individual best_global = population.best();
    int stagnation = 0;

    while (true) {
        for (int generation = 1; generation <= config.max_generations; ++generation) {
            ++total_generations;

            std::size_t p1 = population.tournament_select(config, rng);
            std::size_t p2 = population.tournament_select(config, rng);
            while (p2 == p1) {
                p2 = population.tournament_select(config, rng);
            }

            Individual child = gpx_crossover(
                graph, k,
                population.individuals()[p1],
                population.individuals()[p2],
                rng);

            const TabuColResult tabu = run_tabucol(graph, k, child, config, rng);
            total_tabu_iterations += tabu.iterations_used;

            if (tabu.solved) {
                return {true, 0, total_generations, total_tabu_iterations, child};
            }

            population.replace_candidate(child, config);

            const Individual& best_now = population.best();
            if (best_now.conflict_count < best_global.conflict_count) {
                best_global = best_now;
                stagnation = 0;
            } else {
                ++stagnation;
            }

            if (config.inject_count > 0 && generation % config.inject_every_generations == 0) {
                for (int j = 0; j < config.inject_count; ++j) {
                    const std::uint64_t inject_seed =
                        mix_seed(base_seed, static_cast<std::uint64_t>((k * 100000) + generation * 97 + j + 1));
                    std::mt19937_64 inject_rng(inject_seed);

                    Individual injected = build_initial_candidate(graph, base_order, k, config, inject_seed);
                    const TabuColResult injected_tabu = run_tabucol(graph, k, injected, config, inject_rng);
                    total_tabu_iterations += injected_tabu.iterations_used;

                    if (injected_tabu.solved) {
                        return {true, 0, total_generations, total_tabu_iterations, injected};
                    }

                    population.force_replace_worst_non_elite(injected, config);

                    if (population.best().conflict_count < best_global.conflict_count) {
                        best_global = population.best();
                        stagnation = 0;
                    }
                }
            }

            if (config.verbose && (generation == 1 || generation % 25 == 0)) {
                std::cout
                    << "[k=" << k << "] gen=" << generation
                    << " global_gen=" << total_generations
                    << " best_conflicts=" << best_global.conflict_count
                    << " restarts=" << restart_count
                    << " tabu_iters=" << total_tabu_iterations
                    << '\n';
            }

            if (stagnation >= config.max_stagnation_generations) {
                break;
            }
        }

        if (restart_count >= config.max_restarts) {
            return {false, best_global.conflict_count, total_generations, total_tabu_iterations, best_global};
        }

        ++restart_count;
        stagnation = 0;

        if (config.verbose) {
            std::cout << "[k=" << k << "] restart " << restart_count
                      << " after stagnation, best_conflicts=" << best_global.conflict_count << '\n';
        }

        std::vector<Individual> restarted_population =
            initialize_population(graph, base_order, k, config,
                                  mix_seed(base_seed, static_cast<std::uint64_t>(k * 1000 + restart_count)),
                                  total_tabu_iterations);

        for (const auto& individual : restarted_population) {
            if (individual.conflict_count == 0) {
                return {true, 0, total_generations, total_tabu_iterations, individual};
            }
            if (individual.conflict_count < best_global.conflict_count) {
                best_global = individual;
            }
        }

        population = Population(std::move(restarted_population));
    }
}
}  // namespace

SolveResult solve_graph_coloring(
    const Graph& graph,
    const std::string& instance_path,
    const SolverConfig& config) {

    const std::uint64_t seed = config.use_random_seed ? make_time_seed() : config.seed;
    std::mt19937_64 rng(seed);

    Timer timer;

    const DsaturSeedResult dsatur_seed = build_dsatur_seed(graph, rng);
    const int initial_upper_bound = dsatur_seed.full_coloring.used_colors;

    SolveResult result;
    result.instance_path = instance_path;
    result.initial_upper_bound = initial_upper_bound;
    result.best_feasible_k = initial_upper_bound;
    result.best_colors = dsatur_seed.full_coloring.color;
    result.last_attempt_best_conflicts = 0;

    if (config.verbose) {
        std::cout << "Seed: " << seed << '\n';
        std::cout << "Initial DSATUR upper bound: " << initial_upper_bound << '\n';
    }

    if (config.fixed_k.has_value()) {
        const int k = *config.fixed_k;
        const FixedKRunResult fixed = run_fixed_k(graph, k, config, rng, dsatur_seed.order, seed);

        result.last_attempt_k = k;
        result.last_attempt_solved = fixed.solved;
        result.last_attempt_best_conflicts = fixed.best_conflicts;
        result.total_generations = fixed.generations;
        result.total_tabu_iterations = fixed.tabu_iterations;
        result.elapsed_ms = timer.elapsed_milliseconds();

        if (fixed.solved) {
            result.best_feasible_k = k;
            result.best_colors = fixed.best_individual.color;
        }

        return result;
    }

    if (!config.auto_descend_k) {
        result.last_attempt_k = initial_upper_bound;
        result.last_attempt_solved = true;
        result.last_attempt_best_conflicts = 0;
        result.elapsed_ms = timer.elapsed_milliseconds();
        return result;
    }

    int current_k = initial_upper_bound - 1;
    while (current_k >= 1) {
        if (config.verbose) {
            std::cout << "\nTrying k = " << current_k << '\n';
        }

        const FixedKRunResult run = run_fixed_k(graph, current_k, config, rng, dsatur_seed.order, seed);

        result.total_generations += run.generations;
        result.total_tabu_iterations += run.tabu_iterations;
        result.last_attempt_k = current_k;
        result.last_attempt_solved = run.solved;
        result.last_attempt_best_conflicts = run.best_conflicts;

        if (run.solved) {
            result.best_feasible_k = current_k;
            result.best_colors = run.best_individual.color;
            --current_k;
        } else {
            break;
        }
    }

    result.elapsed_ms = timer.elapsed_milliseconds();
    return result;
}