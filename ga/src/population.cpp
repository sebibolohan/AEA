#include "population.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "diversity.hpp"

namespace {
bool better_than(const Individual& a, const Individual& b) {
    if (a.conflict_count != b.conflict_count) {
        return a.conflict_count < b.conflict_count;
    }
    return a.used_colors < b.used_colors;
}
}  // namespace

Population::Population(std::vector<Individual> individuals)
    : individuals_(std::move(individuals)) {
    if (individuals_.empty()) {
        throw std::invalid_argument("Population cannot be empty");
    }
}

const std::vector<Individual>& Population::individuals() const noexcept {
    return individuals_;
}

std::vector<Individual>& Population::individuals() noexcept {
    return individuals_;
}

std::size_t Population::best_index() const {
    std::size_t idx = 0;
    for (std::size_t i = 1; i < individuals_.size(); ++i) {
        if (better_than(individuals_[i], individuals_[idx])) {
            idx = i;
        }
    }
    return idx;
}

std::size_t Population::worst_index() const {
    std::size_t idx = 0;
    for (std::size_t i = 1; i < individuals_.size(); ++i) {
        if (better_than(individuals_[idx], individuals_[i])) {
            idx = i;
        }
    }
    return idx;
}

const Individual& Population::best() const {
    return individuals_[best_index()];
}

std::vector<std::size_t> Population::rank_indices() const {
    std::vector<std::size_t> idx(individuals_.size());
    for (std::size_t i = 0; i < idx.size(); ++i) {
        idx[i] = i;
    }

    std::sort(idx.begin(), idx.end(), [this](const std::size_t lhs, const std::size_t rhs) {
        return better_than(individuals_[lhs], individuals_[rhs]);
    });

    return idx;
}

std::size_t Population::tournament_select(const SolverConfig& config, std::mt19937_64& rng) const {
    std::uniform_int_distribution<std::size_t> dist(0U, individuals_.size() - 1U);

    std::size_t winner = dist(rng);
    for (int i = 1; i < config.tournament_size; ++i) {
        const std::size_t contender = dist(rng);
        if (better_than(individuals_[contender], individuals_[winner])) {
            winner = contender;
        }
    }
    return winner;
}

void Population::replace_candidate(const Individual& candidate, const SolverConfig& config) {
    if (has_exact_same_coloring(individuals_, candidate)) {
        return;
    }

    const auto ranked = rank_indices();
    const std::size_t elite_keep = static_cast<std::size_t>(config.elite_count);

    std::size_t replace_idx = ranked.back();
    double best_similarity = -1.0;

    for (std::size_t pos = elite_keep; pos < ranked.size(); ++pos) {
        const std::size_t idx = ranked[pos];
        const auto& current = individuals_[idx];

        if (!better_than(candidate, current) && !better_than(current, candidate)) {
            const double distance = greedy_remapped_distance(candidate, current);
            const double similarity = 1.0 - distance;
            if (similarity > best_similarity) {
                best_similarity = similarity;
                replace_idx = idx;
            }
        }
    }

    const double diversity_to_best = greedy_remapped_distance(candidate, individuals_[ranked.front()]);
    const bool candidate_is_better = better_than(candidate, individuals_[replace_idx]);
    const bool sufficiently_diverse = diversity_to_best >= config.min_diversity_threshold;

    if (candidate_is_better || sufficiently_diverse) {
        individuals_[replace_idx] = candidate;
    }
}

void Population::force_replace_worst_non_elite(const Individual& candidate, const SolverConfig& config) {
    const auto ranked = rank_indices();
    const std::size_t elite_keep = static_cast<std::size_t>(config.elite_count);
    const std::size_t idx = ranked[std::max(elite_keep, ranked.size() - 1U)];
    individuals_[idx] = candidate;
}