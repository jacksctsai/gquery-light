#pragma once

#include <cstddef>
#include <iostream>
#include <iomanip>

namespace gq {

/** Per-iteration CUDA event timings for the serialized GPU pipeline (default stream). */
struct PipelineTiming {
    double total_gpu_ms{0.0};
    double h2d_ms{0.0};
    double filter_kernel_ms{0.0};
    double scan_kernel_ms{0.0};
    double scatter_kernel_ms{0.0};
    double atomic_groupby_kernel_ms{0.0};
    double block_partial_reduce_kernel_ms{0.0};
    double final_reduce_kernel_ms{0.0};
    double d2h_ms{0.0};
    /** Host-side merge of per-batch partial GROUP BY results (not GPU time). */
    double cpu_merge_ms{0.0};
};

/** Aggregated averages over measured benchmark iterations (warmup excluded by caller). */
struct TimingStats {
    int count{0};

    double total_gpu_ms{0.0};
    double h2d_ms{0.0};
    double filter_kernel_ms{0.0};
    double scan_kernel_ms{0.0};
    double scatter_kernel_ms{0.0};
    double atomic_groupby_kernel_ms{0.0};
    double block_partial_reduce_kernel_ms{0.0};
    double final_reduce_kernel_ms{0.0};
    double d2h_ms{0.0};
    double cpu_merge_ms{0.0};

    void add(const PipelineTiming& t) {
        ++count;
        total_gpu_ms += t.total_gpu_ms;
        h2d_ms += t.h2d_ms;
        filter_kernel_ms += t.filter_kernel_ms;
        scan_kernel_ms += t.scan_kernel_ms;
        scatter_kernel_ms += t.scatter_kernel_ms;
        atomic_groupby_kernel_ms += t.atomic_groupby_kernel_ms;
        block_partial_reduce_kernel_ms += t.block_partial_reduce_kernel_ms;
        final_reduce_kernel_ms += t.final_reduce_kernel_ms;
        d2h_ms += t.d2h_ms;
        cpu_merge_ms += t.cpu_merge_ms;
    }

    double avg(double sum) const {
        return count > 0 ? sum / static_cast<double>(count) : 0.0;
    }

    double avg_total_gpu_ms() const { return avg(total_gpu_ms); }
    double avg_h2d_ms() const { return avg(h2d_ms); }
    double avg_filter_kernel_ms() const { return avg(filter_kernel_ms); }
    double avg_scan_kernel_ms() const { return avg(scan_kernel_ms); }
    double avg_scatter_kernel_ms() const { return avg(scatter_kernel_ms); }
    double avg_atomic_groupby_kernel_ms() const { return avg(atomic_groupby_kernel_ms); }
    double avg_block_partial_reduce_kernel_ms() const { return avg(block_partial_reduce_kernel_ms); }
    double avg_final_reduce_kernel_ms() const { return avg(final_reduce_kernel_ms); }
    double avg_d2h_ms() const { return avg(d2h_ms); }
    double avg_cpu_merge_ms() const { return avg(cpu_merge_ms); }
};

enum class PipelineTimingMode { Atomic, BlockPartial };

inline void print_pipeline_timing_summary(const TimingStats& stats, PipelineTimingMode mode) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "avg_total_gpu_ms: " << stats.avg_total_gpu_ms() << "\n";
    std::cout << "avg_h2d_ms: " << stats.avg_h2d_ms() << "\n";

    if (mode == PipelineTimingMode::Atomic) {
        std::cout << "avg_atomic_groupby_kernel_ms: " << stats.avg_atomic_groupby_kernel_ms() << "\n";
    } else {
        std::cout << "avg_filter_kernel_ms: " << stats.avg_filter_kernel_ms() << "\n";
        std::cout << "avg_scan_kernel_ms: " << stats.avg_scan_kernel_ms() << "\n";
        std::cout << "avg_scatter_kernel_ms: " << stats.avg_scatter_kernel_ms() << "\n";
        std::cout << "avg_block_partial_reduce_kernel_ms: " << stats.avg_block_partial_reduce_kernel_ms()
                  << "\n";
        if (stats.avg_final_reduce_kernel_ms() > 0.0) {
            std::cout << "avg_final_reduce_kernel_ms: " << stats.avg_final_reduce_kernel_ms() << "\n";
        }
    }

    std::cout << "avg_d2h_ms: " << stats.avg_d2h_ms() << "\n";
    std::cout << "avg_cpu_merge_ms: " << stats.avg_cpu_merge_ms() << "\n";
}

} // namespace gq
