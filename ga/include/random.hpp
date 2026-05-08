#pragma once

#include <chrono>
#include <cstdint>
#include <random>

inline std::uint64_t make_time_seed() {
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return static_cast<std::uint64_t>(now);
}

inline std::mt19937_64 make_rng(std::uint64_t seed) {
    return std::mt19937_64(seed);
}