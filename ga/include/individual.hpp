#pragma once

#include <vector>

struct Individual {
    std::vector<int> color;
    int conflict_count = 0;
    int used_colors = 0;

    [[nodiscard]] bool is_feasible() const noexcept {
        return conflict_count == 0;
    }
};