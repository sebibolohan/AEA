#pragma once

#include <chrono>

class Timer {
public:
    Timer() noexcept : start_(clock_type::now()) {}

    void reset() noexcept {
        start_ = clock_type::now();
    }

    [[nodiscard]] double elapsed_seconds() const noexcept {
        return std::chrono::duration<double>(clock_type::now() - start_).count();
    }

    [[nodiscard]] double elapsed_milliseconds() const noexcept {
        return std::chrono::duration<double, std::milli>(clock_type::now() - start_).count();
    }

private:
    using clock_type = std::chrono::steady_clock;
    clock_type::time_point start_;
};