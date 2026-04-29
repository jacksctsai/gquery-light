#pragma once

#include "gq/types.hpp"
#include <cstdint>

namespace gq {

struct FilterGpuMetrics {
    double h2d_ms{};
    double filter_ms{};
    double scan_ms{};
    double scatter_ms{};
    double d2h_ms{};
    double total_gpu_ms{};

    double effective_h2d_gb_per_s{};
    double filter_gb_per_s{};
    double rows_per_s{};
    double selectivity{};
};

/**
 * Loads columns from host, copies to device, runs one thread per row with atomics for
 * matching rows (same predicate as CPU), copies count/sum back.
 * Metrics are averaged over `iterations` runs.
 */
Result filter_and_sum_gpu_atomic_baseline(const Columns& columns, int iterations, FilterGpuMetrics& metrics,
                                          int threads_per_block = 256);

/**
 * Filter only: produce a 0/1 mask on device. Count is computed by copying the mask back and counting on CPU.
 * Metrics are averaged over `iterations` runs.
 */
FilterGpuMetrics filter_gpu_mask_only(const Columns& columns, int iterations, std::uint64_t& selected_count,
                                      int threads_per_block = 256);

} // namespace gq
