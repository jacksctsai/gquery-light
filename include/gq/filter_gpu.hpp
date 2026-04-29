#pragma once

#include "gq/types.hpp"
#include <cstdint>

namespace gq {

struct FilterGpuMetrics {
    double h2d_ms{};
    double filter_ms{};
    double scan_ms{};
    double scatter_ms{};
    double reduce_ms{};
    double d2h_ms{};
    double total_gpu_ms{};

    double effective_h2d_gb_per_s{};
    double filter_gb_per_s{};
    double rows_per_s{};
    double reduce_rows_per_s{};
    double reduce_gb_per_s{};
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
 * Filter + compact + reduce (atomic selected-sum baseline): build 0/1 mask on device, run prefix-sum,
 * scatter selected row indices, then sum selected fares with global atomic adds. Only final count/sum
 * are copied back.
 * Metrics are averaged over `iterations` runs.
 */
Result filter_and_sum_gpu_compact_atomic(const Columns& columns, int iterations, FilterGpuMetrics& metrics,
                                         int threads_per_block = 256);

/**
 * Filter + compact + reduce (block-partial): each block reduces selected fares into one partial sum,
 * then a second-stage GPU reduction combines partials into the final sum.
 * Metrics are averaged over `iterations` runs.
 */
Result filter_and_sum_gpu_compact_block_partial(const Columns& columns, int iterations, FilterGpuMetrics& metrics,
                                                int threads_per_block = 256);

} // namespace gq
