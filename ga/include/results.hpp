#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct SolveResult {
    std::string instance_path;
    int initial_upper_bound = -1;
    int best_feasible_k = -1;
    int last_attempt_k = -1;
    bool last_attempt_solved = false;
    int last_attempt_best_conflicts = -1;
    int total_generations = 0;
    std::uint64_t total_tabu_iterations = 0;
    double elapsed_ms = 0.0;
    std::vector<int> best_colors;
};

void print_result(const SolveResult& result);
void save_result_json(const SolveResult& result, const std::string& output_path);