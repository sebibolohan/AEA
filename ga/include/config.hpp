#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

struct SolverConfig {
    int population_size = 16;
    int elite_count = 2;
    int tournament_size = 3;

    int max_generations = 2000;
    int max_stagnation_generations = 150;
    int max_restarts = 3;

    int inject_every_generations = 75;
    int inject_count = 2;

    int max_tabucol_iterations = 200000;
    int tabu_tenure_min = 5;
    int tabu_tenure_max = 15;

    int dsatur_order_perturb_swaps = 6;

    double min_diversity_threshold = 0.08;

    bool auto_descend_k = true;
    std::optional<int> fixed_k;

    bool verbose = true;
    bool use_openmp = true;

    std::uint64_t seed = 42;
    bool use_random_seed = false;

    [[nodiscard]] void validate() const {
        if (population_size < 2) {
            throw std::invalid_argument("population_size must be >= 2");
        }
        if (elite_count < 0 || elite_count >= population_size) {
            throw std::invalid_argument("elite_count must be in [0, population_size - 1]");
        }
        if (tournament_size < 2 || tournament_size > population_size) {
            throw std::invalid_argument("tournament_size must be in [2, population_size]");
        }
        if (max_generations <= 0) {
            throw std::invalid_argument("max_generations must be > 0");
        }
        if (max_stagnation_generations <= 0) {
            throw std::invalid_argument("max_stagnation_generations must be > 0");
        }
        if (max_restarts < 0) {
            throw std::invalid_argument("max_restarts must be >= 0");
        }
        if (inject_every_generations <= 0) {
            throw std::invalid_argument("inject_every_generations must be > 0");
        }
        if (inject_count < 0 || inject_count > population_size) {
            throw std::invalid_argument("inject_count must be in [0, population_size]");
        }
        if (max_tabucol_iterations <= 0) {
            throw std::invalid_argument("max_tabucol_iterations must be > 0");
        }
        if (tabu_tenure_min <= 0 || tabu_tenure_max <= 0) {
            throw std::invalid_argument("tabu tenure bounds must be > 0");
        }
        if (tabu_tenure_min > tabu_tenure_max) {
            throw std::invalid_argument("tabu_tenure_min must be <= tabu_tenure_max");
        }
        if (dsatur_order_perturb_swaps < 0) {
            throw std::invalid_argument("dsatur_order_perturb_swaps must be >= 0");
        }
        if (min_diversity_threshold < 0.0 || min_diversity_threshold > 1.0) {
            throw std::invalid_argument("min_diversity_threshold must be in [0, 1]");
        }
        if (fixed_k.has_value() && *fixed_k <= 0) {
            throw std::invalid_argument("fixed_k must be > 0 when provided");
        }
    }
};

struct RunConfig {
    std::string instance_path;
};

struct ProgramOptions {
    RunConfig run;
    SolverConfig solver;
};

inline bool is_flag(std::string_view arg, std::string_view short_name, std::string_view long_name) {
    return arg == short_name || arg == long_name;
}