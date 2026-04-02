#pragma once

#include <chrono>

namespace gq {

class CpuTimer {
public:
    using clock = std::chrono::steady_clock;

    CpuTimer() : start_(clock::now()) {}

    void reset() {
        start_ = clock::now();
    }

    double elapsed_ms() const {
        const auto end = clock::now();
        const auto duration = std::chrono::duration<double, std::milli>(end - start_);
        return duration.count();
    }

private:
    clock::time_point start_;
};

} // namespace gq