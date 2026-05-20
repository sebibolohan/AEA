#include "dimacs_parser.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
[[nodiscard]] std::uint64_t encode_undirected_edge(int u, int v) {
    const auto a = static_cast<std::uint32_t>(std::min(u, v));
    const auto b = static_cast<std::uint32_t>(std::max(u, v));
    return (static_cast<std::uint64_t>(a) << 32U) | static_cast<std::uint64_t>(b);
}
}  // namespace

Graph parse_dimacs_col(const std::string& filepath) {
    std::ifstream fin(filepath);
    if (!fin) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    int declared_n = -1;
    int declared_m = -1;
    bool problem_line_seen = false;

    std::vector<std::pair<int, int>> raw_edges;
    raw_edges.reserve(1024);

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream iss(line);
        char type = '\0';
        iss >> type;

        if (!iss) {
            continue;
        }

        if (type == 'c') {
            continue;
        }

        if (type == 'p') {
            std::string keyword;
            int n = 0;
            int m = 0;
            iss >> keyword >> n >> m;

            if (!iss) {
                throw std::runtime_error("Invalid DIMACS problem line in file: " + filepath);
            }
            if (keyword != "edge" && keyword != "edges") {
                throw std::runtime_error("Unsupported DIMACS problem type: " + keyword);
            }
            if (n <= 0 || m < 0) {
                throw std::runtime_error("Invalid graph size in problem line");
            }

            declared_n = n;
            declared_m = m;
            problem_line_seen = true;
            raw_edges.reserve(static_cast<std::size_t>(m));
            continue;
        }

        if (type == 'e') {
            if (!problem_line_seen) {
                throw std::runtime_error("Encountered edge line before problem line");
            }

            int u_1based = 0;
            int v_1based = 0;
            iss >> u_1based >> v_1based;

            if (!iss) {
                throw std::runtime_error("Invalid edge line in file: " + filepath);
            }
            if (u_1based < 1 || v_1based < 1 || u_1based > declared_n || v_1based > declared_n) {
                throw std::runtime_error("Edge vertex out of range in file: " + filepath);
            }
            if (u_1based == v_1based) {
                continue;
            }

            const int u = u_1based - 1;
            const int v = v_1based - 1;
            raw_edges.emplace_back(u, v);
            continue;
        }
    }

    if (!problem_line_seen) {
        throw std::runtime_error("Missing DIMACS problem line in file: " + filepath);
    }

    std::unordered_set<std::uint64_t> unique_edges;
    unique_edges.reserve(raw_edges.size() * 2U + 1U);

    std::vector<std::pair<int, int>> edges;
    edges.reserve(raw_edges.size());

    for (const auto& [u, v] : raw_edges) {
        const std::uint64_t key = encode_undirected_edge(u, v);
        const auto [_, inserted] = unique_edges.insert(key);
        if (inserted) {
            edges.emplace_back(std::min(u, v), std::max(u, v));
        }
    }

    std::sort(edges.begin(), edges.end());

    Graph graph;
    graph.n = declared_n;
    graph.m = static_cast<int>(edges.size());
    graph.adj.assign(static_cast<std::size_t>(graph.n), {});
    graph.degree.assign(static_cast<std::size_t>(graph.n), 0);
    graph.edges = edges;

    for (const auto& [u, v] : graph.edges) {
        graph.adj[static_cast<std::size_t>(u)].push_back(v);
        graph.adj[static_cast<std::size_t>(v)].push_back(u);
    }

    for (int v = 0; v < graph.n; ++v) {
        auto& nbrs = graph.adj[static_cast<std::size_t>(v)];
        std::sort(nbrs.begin(), nbrs.end());
        graph.degree[static_cast<std::size_t>(v)] = static_cast<int>(nbrs.size());
    }

    graph.validate();

    if (declared_m != static_cast<int>(raw_edges.size())) {
        // intentional no-op:
        // DIMACS files can contain duplicates or oddities; we keep cleaned graph.m
    }

    return graph;
}