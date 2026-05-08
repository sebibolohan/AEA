#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "config.hpp"
#include "dimacs_parser.hpp"
#include "results.hpp"
#include "solver.hpp"
#include "timer.hpp"

namespace {
void print_usage(const char* program_name) {
    std::cout
        << "Usage:\n"
        << "  " << program_name << " --instance <path> [options]\n\n"
        << "Options:\n"
        << "  -i, --instance <path>           Path to DIMACS .col file\n"
        << "  -k, --fixed-k <int>             Run only fixed-k search\n"
        << "      --population <int>          Population size\n"
        << "      --elite <int>               Elite count\n"
        << "      --tournament <int>          Tournament size\n"
        << "      --max-generations <int>     Max generations\n"
        << "      --max-stagnation <int>      Max stagnation generations\n"
        << "      --max-restarts <int>        Max restart count\n"
        << "      --inject-every <int>        Diversity injection frequency\n"
        << "      --inject-count <int>        Number of injected individuals\n"
        << "      --max-tabu-iters <int>      Max TabuCol iterations\n"
        << "      --tabu-min <int>            Min tabu tenure\n"
        << "      --tabu-max <int>            Max tabu tenure\n"
        << "      --order-swaps <int>         DSATUR order perturbation swaps\n"
        << "      --min-diversity <double>    Min diversity threshold\n"
        << "      --seed <uint64>             RNG seed\n"
        << "      --random-seed               Use time-based seed\n"
        << "      --quiet                     Disable verbose output\n"
        << "      --no-openmp                 Disable runtime OpenMP flag\n"
        << "      --no-descend-k              Disable automatic k descent\n"
        << "  -h, --help                      Show help\n";
}

[[nodiscard]] std::string require_value(int argc, char* argv[], int& i, std::string_view flag_name) {
    if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value after " + std::string(flag_name));
    }
    ++i;
    return argv[i];
}

[[nodiscard]] int parse_int_arg(const std::string& value, std::string_view flag_name) {
    try {
        std::size_t pos = 0;
        const int result = std::stoi(value, &pos);
        if (pos != value.size()) {
            throw std::invalid_argument("Trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid integer for " + std::string(flag_name) + ": " + value);
    }
}

[[nodiscard]] std::uint64_t parse_u64_arg(const std::string& value, std::string_view flag_name) {
    try {
        std::size_t pos = 0;
        const auto result = static_cast<std::uint64_t>(std::stoull(value, &pos));
        if (pos != value.size()) {
            throw std::invalid_argument("Trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid uint64 for " + std::string(flag_name) + ": " + value);
    }
}

[[nodiscard]] double parse_double_arg(const std::string& value, std::string_view flag_name) {
    try {
        std::size_t pos = 0;
        const double result = std::stod(value, &pos);
        if (pos != value.size()) {
            throw std::invalid_argument("Trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid double for " + std::string(flag_name) + ": " + value);
    }
}

ProgramOptions parse_cli(int argc, char* argv[]) {
    ProgramOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (is_flag(arg, "-h", "--help")) {
            print_usage(argv[0]);
            std::exit(0);
        } else if (is_flag(arg, "-i", "--instance")) {
            options.run.instance_path = require_value(argc, argv, i, arg);
        } else if (is_flag(arg, "-k", "--fixed-k")) {
            options.solver.fixed_k = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--population") {
            options.solver.population_size = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--elite") {
            options.solver.elite_count = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--tournament") {
            options.solver.tournament_size = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--max-generations") {
            options.solver.max_generations = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--max-stagnation") {
            options.solver.max_stagnation_generations = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--max-restarts") {
            options.solver.max_restarts = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--inject-every") {
            options.solver.inject_every_generations = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--inject-count") {
            options.solver.inject_count = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--max-tabu-iters") {
            options.solver.max_tabucol_iterations = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--tabu-min") {
            options.solver.tabu_tenure_min = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--tabu-max") {
            options.solver.tabu_tenure_max = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--order-swaps") {
            options.solver.dsatur_order_perturb_swaps = parse_int_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--min-diversity") {
            options.solver.min_diversity_threshold = parse_double_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--seed") {
            options.solver.seed = parse_u64_arg(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--random-seed") {
            options.solver.use_random_seed = true;
        } else if (arg == "--quiet") {
            options.solver.verbose = false;
        } else if (arg == "--no-openmp") {
            options.solver.use_openmp = false;
        } else if (arg == "--no-descend-k") {
            options.solver.auto_descend_k = false;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (options.run.instance_path.empty()) {
        throw std::invalid_argument("Missing required argument --instance <path>");
    }

    if (!std::filesystem::exists(options.run.instance_path)) {
        throw std::invalid_argument("Instance file does not exist: " + options.run.instance_path);
    }

    options.solver.validate();
    return options;
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        const ProgramOptions options = parse_cli(argc, argv);

        Timer parse_timer;
        const Graph graph = parse_dimacs_col(options.run.instance_path);
        const double parse_ms = parse_timer.elapsed_milliseconds();

        std::cout << "Instance: " << options.run.instance_path << '\n';
        std::cout << "Vertices: " << graph.n << '\n';
        std::cout << "Edges:    " << graph.m << '\n';
        std::cout << "Parsed in " << parse_ms << " ms\n";

#ifdef GA_USE_OPENMP
        std::cout << "Build OpenMP: enabled\n";
#else
        std::cout << "Build OpenMP: disabled\n";
#endif

        std::cout << "Runtime OpenMP flag: " << (options.solver.use_openmp ? "true" : "false") << '\n';

        const SolveResult result = solve_graph_coloring(graph, options.run.instance_path, options.solver);
        print_result(result);
        save_result_json(result, "results/last_run.json");

        std::cout << "Saved results to results/last_run.json\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}