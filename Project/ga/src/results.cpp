#include "results.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16U);
    for (char ch : s) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += ch;
                break;
        }
    }
    return out;
}
}  // namespace

void print_result(const SolveResult& result) {
    std::cout << "\n=== RESULT ===\n";
    std::cout << "Instance: " << result.instance_path << '\n';
    std::cout << "Initial upper bound: " << result.initial_upper_bound << '\n';
    std::cout << "Best feasible k: " << result.best_feasible_k << '\n';
    std::cout << "Last attempt k: " << result.last_attempt_k << '\n';
    std::cout << "Last attempt solved: " << (result.last_attempt_solved ? "true" : "false") << '\n';
    std::cout << "Last attempt best conflicts: " << result.last_attempt_best_conflicts << '\n';
    std::cout << "Total generations: " << result.total_generations << '\n';
    std::cout << "Total tabu iterations: " << result.total_tabu_iterations << '\n';
    std::cout << "Elapsed ms: " << std::fixed << std::setprecision(3) << result.elapsed_ms << '\n';
}

void save_result_json(const SolveResult& result, const std::string& output_path) {
    const std::filesystem::path path(output_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream fout(output_path);
    if (!fout) {
        throw std::runtime_error("Failed to open result output file: " + output_path);
    }

    fout << "{\n";
    fout << "  \"instance_path\": \"" << escape_json(result.instance_path) << "\",\n";
    fout << "  \"initial_upper_bound\": " << result.initial_upper_bound << ",\n";
    fout << "  \"best_feasible_k\": " << result.best_feasible_k << ",\n";
    fout << "  \"last_attempt_k\": " << result.last_attempt_k << ",\n";
    fout << "  \"last_attempt_solved\": " << (result.last_attempt_solved ? "true" : "false") << ",\n";
    fout << "  \"last_attempt_best_conflicts\": " << result.last_attempt_best_conflicts << ",\n";
    fout << "  \"total_generations\": " << result.total_generations << ",\n";
    fout << "  \"total_tabu_iterations\": " << result.total_tabu_iterations << ",\n";
    fout << "  \"elapsed_ms\": " << std::fixed << std::setprecision(6) << result.elapsed_ms << ",\n";
    fout << "  \"best_colors\": [";

    for (std::size_t i = 0; i < result.best_colors.size(); ++i) {
        if (i > 0U) {
            fout << ", ";
        }
        fout << result.best_colors[i];
    }

    fout << "]\n";
    fout << "}\n";
}