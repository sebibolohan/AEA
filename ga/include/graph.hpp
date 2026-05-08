#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

struct Graph {
    int n = 0;
    int m = 0;
    std::vector<std::vector<int>> adj;
    std::vector<int> degree;
    std::vector<std::pair<int, int>> edges;

    [[nodiscard]] bool empty() const noexcept {
        return n == 0;
    }

    [[nodiscard]] bool valid_vertex(int v) const noexcept {
        return v >= 0 && v < n;
    }

    [[nodiscard]] std::int64_t edge_count_from_adjacency() const {
        std::int64_t total = 0;
        for (const auto& nbrs : adj) {
            total += static_cast<std::int64_t>(nbrs.size());
        }
        return total / 2;
    }

    void validate() const {
        if (n < 0 || m < 0) {
            throw std::runtime_error("Graph has negative n or m");
        }
        if (static_cast<int>(adj.size()) != n) {
            throw std::runtime_error("adj size does not match n");
        }
        if (static_cast<int>(degree.size()) != n) {
            throw std::runtime_error("degree size does not match n");
        }
        for (int v = 0; v < n; ++v) {
            if (degree[v] != static_cast<int>(adj[v].size())) {
                throw std::runtime_error("degree does not match adjacency size");
            }
            for (int u : adj[v]) {
                if (!valid_vertex(u)) {
                    throw std::runtime_error("adjacency contains invalid vertex index");
                }
                if (u == v) {
                    throw std::runtime_error("self-loop detected in adjacency");
                }
            }
        }
        if (static_cast<int>(edges.size()) != m) {
            throw std::runtime_error("edges size does not match m");
        }
    }
};