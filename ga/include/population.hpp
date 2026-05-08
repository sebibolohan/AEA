#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "config.hpp"
#include "individual.hpp"

class Population {
public:
    explicit Population(std::vector<Individual> individuals);

    [[nodiscard]] const std::vector<Individual>& individuals() const noexcept;
    [[nodiscard]] std::vector<Individual>& individuals() noexcept;

    [[nodiscard]] const Individual& best() const;
    [[nodiscard]] std::size_t best_index() const;
    [[nodiscard]] std::size_t worst_index() const;

    [[nodiscard]] std::size_t tournament_select(const SolverConfig& config, std::mt19937_64& rng) const;

    void replace_candidate(const Individual& candidate, const SolverConfig& config);
    void force_replace_worst_non_elite(const Individual& candidate, const SolverConfig& config);

private:
    [[nodiscard]] std::vector<std::size_t> rank_indices() const;
    std::vector<Individual> individuals_;
};