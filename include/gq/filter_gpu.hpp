#pragma once

#include "gq/types.hpp"
#include <cstdint>

namespace gq {

struct GpuRunMetrics {
    double h2d_ms{};
    double kernel_ms{};
    double d2h_ms{};
    double total_gpu_ms{};
    double effective_h2d_gb_per_s{};
};

/**
 * Loads columns from host, copies to device, runs one thread per row with atomics for
 * matching rows (same predicate as CPU), copies count/sum back.
 * Metrics are averaged over `iterations` runs.
 */
Result filter_and_sum_gpu(const Columns& columns, int iterations, GpuRunMetrics& metrics);

} // namespace gq
