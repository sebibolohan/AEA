#include "diversity.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

double normalized_hamming_distance(const Individual& a, const Individual& b) {
    if (a.color.size() != b.color.size()) {
        throw std::invalid_argument("Individuals have different sizes");
    }
    if (a.color.empty()) {
        return 0.0;
    }

    std::size_t diff = 0;
    for (std::size_t i = 0; i < a.color.size(); ++i) {
        if (a.color[i] != b.color[i]) {
            ++diff;
        }
    }

    return static_cast<double>(diff) / static_cast<double>(a.color.size());
}

double greedy_remapped_distance(const Individual& a, const Individual& b) {
    if (a.color.size() != b.color.size()) {
        throw std::invalid_argument("Individuals have different sizes");
    }
    if (a.color.empty()) {
        return 0.0;
    }

    int max_color_a = -1;
    int max_color_b = -1;
    for (const int c : a.color) {
        max_color_a = std::max(max_color_a, c);
    }
    for (const int c : b.color) {
        max_color_b = std::max(max_color_b, c);
    }

    const int ka = max_color_a + 1;
    const int kb = max_color_b + 1;
    if (ka <= 0 || kb <= 0) {
        return 0.0;
    }

    std::vector<std::vector<int>> overlap(
        static_cast<std::size_t>(ka),
        std::vector<int>(static_cast<std::size_t>(kb), 0));

    for (std::size_t i = 0; i < a.color.size(); ++i) {
        const int ca = a.color[i];
        const int cb = b.color[i];
        if (ca >= 0 && cb >= 0) {
            ++overlap[static_cast<std::size_t>(ca)][static_cast<std::size_t>(cb)];
        }
    }

    std::vector<unsigned char> used_a(static_cast<std::size_t>(ka), 0U);
    std::vector<unsigned char> used_b(static_cast<std::size_t>(kb), 0U);

    int matched = 0;
    for (;;) {
        int best_i = -1;
        int best_j = -1;
        int best_overlap = -1;

        for (int i = 0; i < ka; ++i) {
            if (used_a[static_cast<std::size_t>(i)] != 0U) {
                continue;
            }
            for (int j = 0; j < kb; ++j) {
                if (used_b[static_cast<std::size_t>(j)] != 0U) {
                    continue;
                }
                if (overlap[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] > best_overlap) {
                    best_overlap = overlap[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i == -1 || best_j == -1 || best_overlap <= 0) {
            break;
        }

        used_a[static_cast<std::size_t>(best_i)] = 1U;
        used_b[static_cast<std::size_t>(best_j)] = 1U;
        matched += best_overlap;
    }

    return 1.0 - (static_cast<double>(matched) / static_cast<double>(a.color.size()));
}

bool has_exact_same_coloring(const std::vector<Individual>& population, const Individual& candidate) {
    for (const auto& individual : population) {
        if (individual.color == candidate.color) {
            return true;
        }
    }
    return false;
}