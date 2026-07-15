#include <exception>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <string>
#include <filesystem>
#include <map>
#include <random>
#include <vector>

#include "gq/csv_reader.hpp"
#include "gq/filter_cpu.hpp"
#include "gq/timer.hpp"
#include "gq/binary_io.hpp"

#ifdef GQUERY_USE_CUDA
#include "gq/filter_gpu.hpp"
#include "gq/pipeline_timing.hpp"
#include "nvtx_utils.h"
#endif

#ifdef GQUERY_USE_CUDA
/** Distinct buckets in [0, num_groups) with nonzero CPU COUNT after filtering. */
int count_nonempty_groupby_buckets(const gq::GroupByPassengerFilteredResult& gb) {
    int c = 0;
    const std::size_t n = gb.count.size();
    for (std::size_t key = 0; key < n; ++key) {
        if (gb.count[key] != 0U) {
            ++c;
        }
    }
    return c;
}

void print_groupby_per_key_correctness(const gq::GroupByPassengerFilteredResult& cpu, const gq::GroupByGpuTable& gpu) {
    constexpr double kAbsEps = 1e-9;
    constexpr double kRelEps = 1e-9;

    if (cpu.count.size() != gpu.count.size() || cpu.sum.size() != gpu.sum.size()) {
        std::cout << "error: CPU/GPU GROUP BY shape mismatch\n";
        return;
    }

    const std::size_t num_groups = cpu.count.size();
    std::cout << "\n--- Per-key GROUP BY (keys in [0, " << (num_groups == 0 ? 0 : (num_groups - 1)) << "]) ---\n";
    std::cout << std::setw(6) << "key" << " | " << std::setw(10) << "cpu_count" << " " << std::setw(10) << "gpu_count"
              << " " << std::setw(12) << "count_delta" << " | " << std::setw(14) << "cpu_sum" << " " << std::setw(14)
              << "gpu_sum" << " " << std::setw(14) << "sum_delta" << " | " << std::setw(8) << "correct" << "\n";

    bool all_ok = true;
    for (std::size_t b = 0; b < num_groups; ++b) {
        const std::uint64_t cc = cpu.count[b];
        const std::uint64_t gc = gpu.count[b];
        const double cs = cpu.sum[b];
        const double gs = gpu.sum[b];

        const std::int64_t count_delta = static_cast<std::int64_t>(gc) - static_cast<std::int64_t>(cc);
        const double sum_delta = gs - cs;
        const double sum_tol = kAbsEps + kRelEps * std::max(std::abs(cs), 1.0);

        const bool count_ok = (cc == gc);
        const bool sum_ok = std::abs(sum_delta) <= sum_tol;
        const bool row_ok = count_ok && sum_ok;
        if (!row_ok) {
            all_ok = false;
        }

        std::cout << std::setw(6) << b << " | " << std::setw(10) << cc << " " << std::setw(10) << gc << " "
                  << std::setw(12) << count_delta << " | " << std::setw(14) << std::fixed << std::setprecision(7) << cs
                  << " " << std::setw(14) << gs << " " << std::setw(14) << sum_delta << " | " << std::setw(8)
                  << (row_ok ? "1" : "0") << "\n";
    }

    std::cout << "all_keys_correct: " << (all_ok ? "1" : "0") << "\n";
    std::cout << "---------------------------------------------------------------\n";
}

void print_runbin_gpu_groupby_metrics(std::uint64_t file_bytes, std::uint64_t rows, int iterations, double load_ms,
                                      int threads_per_block, const gq::FilterGpuMetrics& gpu, double total_ms,
                                      const std::string& mode_csv, const std::string& groupby_algorithm_label) {
    const std::uint64_t payload_bytes = rows * (2 * sizeof(float) + sizeof(std::uint32_t));
    const double load_seconds = load_ms / 1000.0;
    const double load_gb_per_s = load_seconds > 0.0 ? (static_cast<double>(file_bytes) / load_seconds / 1e9) : 0.0;

    std::cout << "file_bytes: " << file_bytes << "\n";
    std::cout << "payload_bytes: " << payload_bytes << "\n";
    std::cout << "rows: " << rows << "\n";
    std::cout << "iterations: " << iterations << "\n";
    std::cout << "threads_per_block: " << threads_per_block << "\n";
    std::cout << "load_ms: " << std::fixed << std::setprecision(4) << load_ms << "\n";
    std::cout << "h2d_ms: " << std::fixed << std::setprecision(4) << gpu.h2d_ms << "\n";
    std::cout << "filter_ms: " << gpu.filter_ms << "\n";
    std::cout << "scan_ms: " << gpu.scan_ms << "\n";
    std::cout << "scatter_ms: " << gpu.scatter_ms << "\n";
    std::cout << "reduce_ms: " << gpu.reduce_ms << "\n";
    std::cout << "groupby_ms: " << gpu.groupby_ms << "\n";
    std::cout << "d2h_ms: " << gpu.d2h_ms << "\n";
    std::cout << "total_gpu_ms: " << gpu.total_gpu_ms << "\n";
    std::cout << "total_ms: " << total_ms << "\n";
    std::cout << "effective_h2d_gb_per_s: " << gpu.effective_h2d_gb_per_s << "\n";
    std::cout << "filter_gb_per_s: " << gpu.filter_gb_per_s << "\n";
    std::cout << "rows_per_s: " << gpu.rows_per_s << "\n";
    std::cout << "groupby_rows_per_s: " << gpu.groupby_rows_per_s << "\n";
    std::cout << "groupby_gb_per_s: " << gpu.groupby_gb_per_s << "\n";
    std::cout << "selectivity: " << gpu.selectivity << "\n";
    std::cout << "selected_count: " << gpu.predicate_selected_count << "\n";
    std::cout << "key_cardinality: " << gpu.key_cardinality << "\n";
    std::cout << "max_key: " << gpu.max_key_observed << "\n";
    std::cout << "groupby_algorithm: " << groupby_algorithm_label << "\n";
    std::cout << "load_gb_per_s: " << load_gb_per_s << "\n";

    std::cout << "mode=" << mode_csv << ",layout=soa,"
              << "file_bytes=" << file_bytes << ","
              << "payload_bytes=" << payload_bytes << ","
              << "rows=" << rows << ","
              << "iterations=" << iterations << ","
              << "threads_per_block=" << threads_per_block << ","
              << "load_ms=" << load_ms << ","
              << "h2d_ms=" << gpu.h2d_ms << ","
              << "filter_ms=" << gpu.filter_ms << ","
              << "scan_ms=" << gpu.scan_ms << ","
              << "scatter_ms=" << gpu.scatter_ms << ","
              << "reduce_ms=" << gpu.reduce_ms << ","
              << "groupby_ms=" << gpu.groupby_ms << ","
              << "d2h_ms=" << gpu.d2h_ms << ","
              << "total_gpu_ms=" << gpu.total_gpu_ms << ","
              << "total_ms=" << total_ms << ","
              << "effective_h2d_gb_per_s=" << gpu.effective_h2d_gb_per_s << ","
              << "filter_gb_per_s=" << gpu.filter_gb_per_s << ","
              << "rows_per_s=" << gpu.rows_per_s << ","
              << "groupby_rows_per_s=" << gpu.groupby_rows_per_s << ","
              << "groupby_gb_per_s=" << gpu.groupby_gb_per_s << ","
              << "selectivity=" << gpu.selectivity << ","
              << "selected_count=" << gpu.predicate_selected_count << ","
              << "key_cardinality=" << gpu.key_cardinality << ","
              << "max_key=" << gpu.max_key_observed << ","
              << "groupby_algorithm=" << groupby_algorithm_label << ","
              << "load_gb_per_s=" << load_gb_per_s << "\n";
}
#endif

void print_cardinality(const gq::CardinalityResult& card) {
    std::cout << "\n--- Passenger Count Cardinality ---\n";
    std::cout << "Distinct Keys: " << card.counts.size() << "\n";
    if (!card.counts.empty()) {
        std::cout << "Min Key: " << card.min_key << "\n";
        std::cout << "Max Key: " << card.max_key << "\n";
        std::cout << "Counts:\n";
        for (const auto& [k, v] : card.counts) {
            std::cout << "  " << k << ": " << v << "\n";
        }
    }
    std::cout << "-----------------------------------\n";
}

void print_metrics(const std::string& mode, const std::string& layout, std::uint64_t file_bytes, std::uint64_t rows, int iterations, double load_ms, double filter_ms_avg, double total_ms, std::uint64_t count, double sum_fare_amount, double parse_mb_per_s) {
    
    std::uint64_t payload_bytes = rows * (2 * sizeof(float) + sizeof(std::uint8_t));
    double filter_seconds_avg = filter_ms_avg / 1000.0;
    double filter_gb_per_s = filter_seconds_avg > 0.0 ? (static_cast<double>(payload_bytes) / filter_seconds_avg / 1e9) : 0.0;
    
    double load_seconds = load_ms / 1000.0;
    double load_gb_per_s = load_seconds > 0.0 ? (static_cast<double>(file_bytes) / load_seconds / 1e9) : 0.0;

    std::cout << "file_bytes: " << file_bytes << "\n";
    std::cout << "payload_bytes: " << payload_bytes << "\n";
    std::cout << "rows: " << rows << "\n";
    std::cout << "iterations: " << iterations << "\n";
    std::cout << "load_ms: " << std::fixed << std::setprecision(4) << load_ms << "\n";
    std::cout << "filter_ms (avg): " << std::fixed << std::setprecision(4) << filter_ms_avg << "\n";
    std::cout << "total_ms: " << std::fixed << std::setprecision(4) << total_ms << "\n";
    std::cout << "count: " << count << "\n";
    std::cout << "sum_fare_amount: " << std::fixed << std::setprecision(4) << sum_fare_amount << "\n";
    if (parse_mb_per_s > 0) {
        std::cout << "parse_mb_per_s: " << std::fixed << std::setprecision(4) << parse_mb_per_s << "\n";
    }
    if (mode == "runbin") {
        std::cout << "load_gb_per_s: " << std::fixed << std::setprecision(4) << load_gb_per_s << "\n";
    }
    std::cout << "filter_gb_per_s: " << std::fixed << std::setprecision(4) << filter_gb_per_s << "\n";

    // Machine-readable line (CPU-oriented)
    std::cout << "mode=" << mode << ",layout=" << layout << ","
              << "file_bytes=" << file_bytes << ","
              << "payload_bytes=" << payload_bytes << ","
              << "rows=" << rows << ","
              << "iterations=" << iterations << ","
              << "load_ms=" << std::fixed << std::setprecision(4) << load_ms << ","
              << "filter_ms=" << filter_ms_avg << ","
              << "total_ms=" << total_ms << ","
              << "count=" << count << ","
              << "sum_fare_amount=" << sum_fare_amount << ","
              << "parse_mb_per_s=" << parse_mb_per_s << ","
              << "load_gb_per_s=" << load_gb_per_s << ","
              << "filter_gb_per_s=" << filter_gb_per_s << "\n";
}

#ifdef GQUERY_USE_CUDA
void print_runbin_gpu_metrics(std::uint64_t file_bytes, std::uint64_t rows, int iterations, double load_ms,
                              int threads_per_block, const gq::FilterGpuMetrics& gpu, double total_ms, std::uint64_t count,
                              double sum_fare_amount) {
    const std::uint64_t payload_bytes = rows * (2 * sizeof(float) + sizeof(std::uint8_t));
    const double load_seconds = load_ms / 1000.0;
    const double load_gb_per_s = load_seconds > 0.0 ? (static_cast<double>(file_bytes) / load_seconds / 1e9) : 0.0;

    std::cout << "file_bytes: " << file_bytes << "\n";
    std::cout << "payload_bytes: " << payload_bytes << "\n";
    std::cout << "rows: " << rows << "\n";
    std::cout << "iterations: " << iterations << "\n";
    std::cout << "threads_per_block: " << threads_per_block << "\n";
    std::cout << "load_ms: " << std::fixed << std::setprecision(4) << load_ms << "\n";
    std::cout << "h2d_ms: " << std::fixed << std::setprecision(4) << gpu.h2d_ms << "\n";
    std::cout << "filter_ms: " << std::fixed << std::setprecision(4) << gpu.filter_ms << "\n";
    std::cout << "scan_ms: " << std::fixed << std::setprecision(4) << gpu.scan_ms << "\n";
    std::cout << "scatter_ms: " << std::fixed << std::setprecision(4) << gpu.scatter_ms << "\n";
    std::cout << "reduce_ms: " << std::fixed << std::setprecision(4) << gpu.reduce_ms << "\n";
    std::cout << "d2h_ms: " << std::fixed << std::setprecision(4) << gpu.d2h_ms << "\n";
    std::cout << "total_gpu_ms: " << std::fixed << std::setprecision(4) << gpu.total_gpu_ms << "\n";
    std::cout << "total_ms: " << std::fixed << std::setprecision(4) << total_ms << "\n";
    std::cout << "effective_h2d_gb_per_s: " << std::fixed << std::setprecision(4) << gpu.effective_h2d_gb_per_s
              << "\n";
    std::cout << "filter_gb_per_s: " << std::fixed << std::setprecision(4) << gpu.filter_gb_per_s << "\n";
    std::cout << "rows_per_s: " << std::fixed << std::setprecision(2) << gpu.rows_per_s << "\n";
    std::cout << "selectivity: " << std::fixed << std::setprecision(6) << gpu.selectivity << "\n";
    std::cout << "load_gb_per_s: " << std::fixed << std::setprecision(4) << load_gb_per_s << "\n";
    std::cout << "count: " << count << "\n";
    std::cout << "sum_fare_amount: " << std::fixed << std::setprecision(4) << sum_fare_amount << "\n";

    std::cout << "mode=gpu_atomic_baseline,layout=soa,"
              << "file_bytes=" << file_bytes << ","
              << "payload_bytes=" << payload_bytes << ","
              << "rows=" << rows << ","
              << "iterations=" << iterations << ","
              << "threads_per_block=" << threads_per_block << ","
              << "load_ms=" << std::fixed << std::setprecision(4) << load_ms << ","
              << "h2d_ms=" << gpu.h2d_ms << ","
              << "filter_ms=" << gpu.filter_ms << ","
              << "scan_ms=" << gpu.scan_ms << ","
              << "scatter_ms=" << gpu.scatter_ms << ","
              << "reduce_ms=" << gpu.reduce_ms << ","
              << "d2h_ms=" << gpu.d2h_ms << ","
              << "total_gpu_ms=" << gpu.total_gpu_ms << ","
              << "total_ms=" << total_ms << ","
              << "effective_h2d_gb_per_s=" << gpu.effective_h2d_gb_per_s << ","
              << "filter_gb_per_s=" << gpu.filter_gb_per_s << ","
              << "rows_per_s=" << gpu.rows_per_s << ","
              << "selectivity=" << gpu.selectivity << ","
              << "load_gb_per_s=" << load_gb_per_s << ","
              << "count=" << count << ","
              << "sum_fare_amount=" << sum_fare_amount << "\n";
}

void print_runbin_gpu_mask_metrics(std::uint64_t file_bytes, std::uint64_t rows, int iterations, double load_ms,
                                   int threads_per_block, const gq::FilterGpuMetrics& gpu, double total_ms,
                                   std::uint64_t selected_count, double sum_fare_amount,
                                   const std::string& reduce_algorithm) {
    const std::uint64_t payload_bytes = rows * 2 * sizeof(float);
    const double load_seconds = load_ms / 1000.0;
    const double load_gb_per_s = load_seconds > 0.0 ? (static_cast<double>(file_bytes) / load_seconds / 1e9) : 0.0;

    std::cout << "file_bytes: " << file_bytes << "\n";
    std::cout << "payload_bytes: " << payload_bytes << "\n";
    std::cout << "rows: " << rows << "\n";
    std::cout << "iterations: " << iterations << "\n";
    std::cout << "threads_per_block: " << threads_per_block << "\n";
    std::cout << "load_ms: " << std::fixed << std::setprecision(4) << load_ms << "\n";
    std::cout << "h2d_ms: " << std::fixed << std::setprecision(4) << gpu.h2d_ms << "\n";
    std::cout << "filter_ms: " << std::fixed << std::setprecision(4) << gpu.filter_ms << "\n";
    std::cout << "scan_ms: " << std::fixed << std::setprecision(4) << gpu.scan_ms << "\n";
    std::cout << "scatter_ms: " << std::fixed << std::setprecision(4) << gpu.scatter_ms << "\n";
    std::cout << "reduce_ms: " << std::fixed << std::setprecision(4) << gpu.reduce_ms << "\n";
    std::cout << "d2h_ms: " << std::fixed << std::setprecision(4) << gpu.d2h_ms << "\n";
    std::cout << "total_gpu_ms: " << std::fixed << std::setprecision(4) << gpu.total_gpu_ms << "\n";
    std::cout << "total_ms: " << std::fixed << std::setprecision(4) << total_ms << "\n";
    std::cout << "effective_h2d_gb_per_s: " << std::fixed << std::setprecision(4) << gpu.effective_h2d_gb_per_s
              << "\n";
    std::cout << "filter_gb_per_s: " << std::fixed << std::setprecision(4) << gpu.filter_gb_per_s << "\n";
    std::cout << "rows_per_s: " << std::fixed << std::setprecision(2) << gpu.rows_per_s << "\n";
    std::cout << "reduce_rows_per_s: " << std::fixed << std::setprecision(2) << gpu.reduce_rows_per_s << "\n";
    std::cout << "reduce_gb_per_s: " << std::fixed << std::setprecision(4) << gpu.reduce_gb_per_s << "\n";
    std::cout << "selectivity: " << std::fixed << std::setprecision(6) << gpu.selectivity << "\n";
    std::cout << "selected_count: " << selected_count << "\n";
    std::cout << "sum_fare_amount: " << std::fixed << std::setprecision(4) << sum_fare_amount << "\n";
    std::cout << "reduce_algorithm: " << reduce_algorithm << "\n";
    std::cout << "load_gb_per_s: " << std::fixed << std::setprecision(4) << load_gb_per_s << "\n";

    std::cout << "mode=gpu_mask,layout=soa,"
              << "file_bytes=" << file_bytes << ","
              << "payload_bytes=" << payload_bytes << ","
              << "rows=" << rows << ","
              << "iterations=" << iterations << ","
              << "threads_per_block=" << threads_per_block << ","
              << "load_ms=" << std::fixed << std::setprecision(4) << load_ms << ","
              << "h2d_ms=" << gpu.h2d_ms << ","
              << "filter_ms=" << gpu.filter_ms << ","
              << "scan_ms=" << gpu.scan_ms << ","
              << "scatter_ms=" << gpu.scatter_ms << ","
              << "reduce_ms=" << gpu.reduce_ms << ","
              << "d2h_ms=" << gpu.d2h_ms << ","
              << "total_gpu_ms=" << gpu.total_gpu_ms << ","
              << "total_ms=" << total_ms << ","
              << "effective_h2d_gb_per_s=" << gpu.effective_h2d_gb_per_s << ","
              << "filter_gb_per_s=" << gpu.filter_gb_per_s << ","
              << "rows_per_s=" << gpu.rows_per_s << ","
              << "reduce_rows_per_s=" << gpu.reduce_rows_per_s << ","
              << "reduce_gb_per_s=" << gpu.reduce_gb_per_s << ","
              << "selectivity=" << gpu.selectivity << ","
              << "selected_count=" << selected_count << ","
              << "sum_fare_amount=" << sum_fare_amount << ","
              << "reduce_algorithm=" << reduce_algorithm << ","
              << "load_gb_per_s=" << load_gb_per_s << "\n";
}

void print_cpu_gpu_correctness(const gq::Result& cpu, const gq::Result& gpu) {
    const std::uint64_t cpu_count = cpu.count;
    const std::uint64_t gpu_count = gpu.count;
    const double cpu_sum = cpu.sum_fare_amount;
    const double gpu_sum = gpu.sum_fare_amount;

    const std::int64_t count_delta = static_cast<std::int64_t>(gpu_count) - static_cast<std::int64_t>(cpu_count);
    const double sum_delta = gpu_sum - cpu_sum;
    const double abs_sum_delta = std::abs(sum_delta);

    constexpr double kAbsEps = 1e-9;
    constexpr double kRelEps = 1e-9;
    const double sum_tol = kAbsEps + kRelEps * std::max(std::abs(cpu_sum), 1.0);
    const bool count_ok = (cpu_count == gpu_count);
    const bool sum_ok = (abs_sum_delta <= sum_tol);

    std::cout << "cpu_count: " << cpu_count << "\n";
    std::cout << "gpu_count: " << gpu_count << "\n";
    std::cout << "count_delta: " << count_delta << "\n";
    std::cout << "cpu_sum_fare_amount: " << std::fixed << std::setprecision(8) << cpu_sum << "\n";
    std::cout << "gpu_sum_fare_amount: " << std::fixed << std::setprecision(8) << gpu_sum << "\n";
    std::cout << "sum_delta: " << std::fixed << std::setprecision(12) << sum_delta << "\n";
    std::cout << "abs_sum_delta: " << std::fixed << std::setprecision(12) << abs_sum_delta << "\n";
    std::cout << "sum_tol: " << std::fixed << std::setprecision(12) << sum_tol << "\n";
    std::cout << "correct_count: " << (count_ok ? 1 : 0) << "\n";
    std::cout << "correct_sum: " << (sum_ok ? 1 : 0) << "\n";
}

#endif

enum class Distribution { Uniform, Hotkey, Zipf };
enum class ReductionMode { Atomic, BlockPartial };

struct BenchmarkConfig {
    std::size_t num_rows = 12'000'000;
    int num_groups = 1024;
    Distribution distribution = Distribution::Uniform;
    ReductionMode mode = ReductionMode::BlockPartial;
    int warmup = 3;
    int iterations = 10;
    int seed = 42;
    bool validate = true;
    int threads_per_block = 256;
    gq::HostMemoryMode memory = gq::HostMemoryMode::Pageable;
    gq::ExecutionMode execution = gq::ExecutionMode::Sync;
    std::size_t batch_rows = 0;
};

static std::size_t compute_num_batches(std::size_t num_rows, std::size_t batch_rows) {
    if (batch_rows == 0 || batch_rows >= num_rows) {
        return 1;
    }
    return (num_rows + batch_rows - 1) / batch_rows;
}

static const char* to_string(Distribution d) {
    switch (d) {
        case Distribution::Uniform: return "uniform";
        case Distribution::Hotkey: return "hotkey";
        case Distribution::Zipf: return "zipf";
    }
    return "unknown";
}

static const char* to_string(ReductionMode m) {
    switch (m) {
        case ReductionMode::Atomic: return "atomic";
        case ReductionMode::BlockPartial: return "block-partial";
    }
    return "unknown";
}

static const char* to_string(gq::HostMemoryMode m) {
    switch (m) {
        case gq::HostMemoryMode::Pageable: return "pageable";
        case gq::HostMemoryMode::Pinned: return "pinned";
    }
    return "unknown";
}

static const char* to_string(gq::ExecutionMode e) {
    switch (e) {
        case gq::ExecutionMode::Sync: return "sync";
        case gq::ExecutionMode::SingleStreamAsync: return "single-stream-async";
    }
    return "unknown";
}

static void print_usage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0 << " [options]\n"
              << "  " << argv0 << " csv2bin <input.csv> <output.bin>\n"
              << "  " << argv0 << " runbin <input.bin> <iterations>\n"
              << "  " << argv0 << " runbin_gpu_atomic <input.bin> [iterations] [threads_per_block]\n"
              << "  " << argv0 << " runbin_gpu_groupby_atomic <input.bin> [iterations] [threads_per_block]\n"
              << "  " << argv0 << " runbin_gpu_groupby_block_partial <input.bin> [iterations] [threads_per_block]\n"
              << "  " << argv0 << " runbin_gpu_mask_atomic <input.bin> [iterations] [threads_per_block]\n"
              << "  " << argv0 << " runbin_gpu_mask_block_partial <input.bin> [iterations] [threads_per_block]\n"
              << "  " << argv0 << " runbin_gpu_mask <input.bin> [iterations] [threads_per_block]\n"
              << "  " << argv0 << " <input.csv> [iterations]\n\n"
              << "Options (synthetic GPU GROUP BY benchmark):\n"
              << "  --num-rows N                 Number of logical input rows\n"
              << "  --num-groups N               Number of group keys\n"
              << "  --distribution uniform|hotkey|zipf\n"
              << "  --mode atomic|block-partial\n"
              << "  --warmup N\n"
              << "  --iterations N\n"
              << "  --seed N\n"
              << "  --validate true|false\n"
              << "  --threads-per-block N\n"
              << "  --memory pageable|pinned\n"
              << "  --execution sync|single-stream-async\n"
              << "  --batch-rows N             Contiguous row-batch size (0 = all rows)\n"
              << "  --help\n";
}

static bool parse_bool(const std::string& s) {
    if (s == "true" || s == "1") return true;
    if (s == "false" || s == "0") return false;
    throw std::runtime_error("invalid bool: " + s);
}

static BenchmarkConfig parseArgs(int argc, char** argv) {
    BenchmarkConfig cfg{};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--num-rows") {
            cfg.num_rows = static_cast<std::size_t>(std::stoull(need_value("--num-rows")));
        } else if (arg == "--num-groups") {
            cfg.num_groups = std::stoi(need_value("--num-groups"));
        } else if (arg == "--distribution") {
            const std::string v = need_value("--distribution");
            if (v == "uniform") cfg.distribution = Distribution::Uniform;
            else if (v == "hotkey") cfg.distribution = Distribution::Hotkey;
            else if (v == "zipf") cfg.distribution = Distribution::Zipf;
            else throw std::runtime_error("unknown --distribution: " + v);
        } else if (arg == "--mode") {
            const std::string v = need_value("--mode");
            if (v == "atomic") cfg.mode = ReductionMode::Atomic;
            else if (v == "block-partial") cfg.mode = ReductionMode::BlockPartial;
            else throw std::runtime_error("unknown --mode: " + v);
        } else if (arg == "--warmup") {
            cfg.warmup = std::stoi(need_value("--warmup"));
        } else if (arg == "--iterations") {
            cfg.iterations = std::stoi(need_value("--iterations"));
        } else if (arg == "--seed") {
            cfg.seed = std::stoi(need_value("--seed"));
        } else if (arg == "--validate") {
            cfg.validate = parse_bool(need_value("--validate"));
        } else if (arg == "--threads-per-block") {
            cfg.threads_per_block = std::stoi(need_value("--threads-per-block"));
        } else if (arg == "--memory") {
            const std::string v = need_value("--memory");
            if (v == "pageable") {
                cfg.memory = gq::HostMemoryMode::Pageable;
            } else if (v == "pinned") {
                cfg.memory = gq::HostMemoryMode::Pinned;
            } else {
                throw std::runtime_error("unknown --memory: " + v);
            }
        } else if (arg == "--execution") {
            const std::string v = need_value("--execution");
            if (v == "sync") {
                cfg.execution = gq::ExecutionMode::Sync;
            } else if (v == "single-stream-async") {
                cfg.execution = gq::ExecutionMode::SingleStreamAsync;
            } else {
                throw std::runtime_error("unknown --execution: " + v);
            }
        } else if (arg == "--batch-rows") {
            const std::string v = need_value("--batch-rows");
            if (!v.empty() && v[0] == '-') {
                throw std::runtime_error("--batch-rows must be >= 0");
            }
            const unsigned long long raw = std::stoull(v);
            cfg.batch_rows = static_cast<std::size_t>(raw);
            if (static_cast<unsigned long long>(cfg.batch_rows) != raw) {
                throw std::runtime_error("--batch-rows value overflows size_t");
            }
        } else if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (cfg.num_rows == 0) throw std::runtime_error("--num-rows must be > 0");
    if (cfg.num_groups <= 0) throw std::runtime_error("--num-groups must be > 0");
    if (cfg.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
    if (cfg.iterations <= 0) throw std::runtime_error("--iterations must be > 0");
    if (cfg.threads_per_block <= 0) throw std::runtime_error("--threads-per-block must be > 0");
    return cfg;
}

static void printConfig(const BenchmarkConfig& cfg) {
    std::cout << "G-Query Light Benchmark Config\n";
    std::cout << "num_rows:     " << cfg.num_rows << "\n";
    std::cout << "num_groups:   " << cfg.num_groups << "\n";
    std::cout << "distribution: " << to_string(cfg.distribution) << "\n";
    std::cout << "mode:         " << to_string(cfg.mode) << "\n";
    std::cout << "warmup:       " << cfg.warmup << "\n";
    std::cout << "iterations:   " << cfg.iterations << "\n";
    std::cout << "seed:         " << cfg.seed << "\n";
    std::cout << "validate:     " << (cfg.validate ? "true" : "false") << "\n";
    std::cout << "threads_per_block: " << cfg.threads_per_block << "\n";
    std::cout << "memory:       " << to_string(cfg.memory) << "\n";
    std::cout << "execution:    " << to_string(cfg.execution) << "\n";
    std::cout << "batch_rows:   " << cfg.batch_rows << "\n";
    std::cout << "num_batches:  " << compute_num_batches(cfg.num_rows, cfg.batch_rows) << "\n";
}

static gq::Columns generateDataset(const BenchmarkConfig& cfg) {
    gq::Columns cols;
    const bool pinned = cfg.memory == gq::HostMemoryMode::Pinned;
    cols.passenger_count.resize(cfg.num_rows, pinned);
    cols.trip_distance.resize(cfg.num_rows, pinned);
    cols.fare_amount.resize(cfg.num_rows, pinned);

    std::mt19937 rng(static_cast<std::uint32_t>(cfg.seed));
    std::uniform_real_distribution<float> td_dist(0.0f, 6.0f);
    std::uniform_real_distribution<float> fa_dist(0.0f, 25.0f);
    std::uniform_int_distribution<std::uint32_t> key_all(0U, static_cast<std::uint32_t>(cfg.num_groups - 1));
    std::uniform_int_distribution<std::uint32_t> key_cold(1U, static_cast<std::uint32_t>(std::max(cfg.num_groups - 1, 1)));
    std::uniform_real_distribution<float> p01(0.0f, 1.0f);

    for (std::size_t i = 0; i < cfg.num_rows; ++i) {
        cols.trip_distance[i] = td_dist(rng);
        cols.fare_amount[i] = fa_dist(rng);
        std::uint32_t k = 0;
        switch (cfg.distribution) {
            case Distribution::Uniform:
                k = key_all(rng);
                break;
            case Distribution::Hotkey:
                k = (p01(rng) < 0.90f) ? 0U : key_cold(rng);
                break;
            case Distribution::Zipf:
                throw std::runtime_error("zipf distribution not implemented yet");
        }
        cols.passenger_count[i] = k;
    }
    return cols;
}

int main(int argc, char** argv) {
    try {
        // Synthetic benchmark mode: any --options triggers config parsing.
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]).rfind("--", 0) == 0) {
#ifndef GQUERY_USE_CUDA
                std::cerr << "error: CUDA not enabled in this build\n";
                return 1;
#else
                BenchmarkConfig cfg = parseArgs(argc, argv);
                NvtxRange program_range("program_total");
                printConfig(cfg);

                if (cfg.execution == gq::ExecutionMode::SingleStreamAsync &&
                    cfg.memory == gq::HostMemoryMode::Pageable) {
                    std::cerr
                        << "Warning: single-stream-async with pageable memory may not provide true async host "
                           "transfer behavior. Use --memory pinned for async transfer experiments.\n";
                }

                gq::Columns dataset;
                double gen_ms = 0.0;
                {
                    NvtxRange gen_range("cpu_generate_input");
                    const NvtxRange memory_range(cfg.memory == gq::HostMemoryMode::Pinned ? "host_memory_pinned"
                                                                                          : "host_memory_pageable");
                    gq::CpuTimer gen_timer;
                    dataset = generateDataset(cfg);
                    gen_ms = gen_timer.elapsed_ms();
                }
                std::cout << "data_generation_ms: " << std::fixed << std::setprecision(4) << gen_ms << "\n";

                // Warmup (not included in measured metrics)
                if (cfg.warmup > 0) {
                    NvtxRange warmup_range("benchmark_warmup");
                    gq::FilterGpuMetrics warm_metrics{};
                    if (cfg.mode == ReductionMode::Atomic) {
                        (void)gq::filter_groupby_gpu_atomic_baseline(dataset, cfg.warmup, warm_metrics,
                                                                     static_cast<std::size_t>(cfg.num_groups),
                                                                     cfg.threads_per_block, "warmup_iteration",
                                                                     nullptr, cfg.execution, cfg.batch_rows);
                    } else {
                        (void)gq::filter_groupby_gpu_compact_block_partial(dataset, cfg.warmup, warm_metrics,
                                                                           static_cast<std::size_t>(cfg.num_groups),
                                                                           cfg.threads_per_block, "warmup_iteration",
                                                                           nullptr, cfg.execution, cfg.batch_rows);
                    }
                }

                // Measured iterations
                gq::FilterGpuMetrics gpu_metrics{};
                gq::TimingStats timing_stats{};
                gq::GroupByGpuTable gpu_gb{};
                const gq::PipelineTimingMode timing_mode =
                    (cfg.mode == ReductionMode::Atomic) ? gq::PipelineTimingMode::Atomic
                                                        : gq::PipelineTimingMode::BlockPartial;
                {
                    NvtxRange measured_range("benchmark_measured_iterations");
                    if (cfg.mode == ReductionMode::Atomic) {
                        gpu_gb = gq::filter_groupby_gpu_atomic_baseline(dataset, cfg.iterations, gpu_metrics,
                                                                        static_cast<std::size_t>(cfg.num_groups),
                                                                        cfg.threads_per_block, "measured_iteration",
                                                                        &timing_stats, cfg.execution, cfg.batch_rows);
                    } else {
                        gpu_gb = gq::filter_groupby_gpu_compact_block_partial(
                            dataset, cfg.iterations, gpu_metrics, static_cast<std::size_t>(cfg.num_groups),
                            cfg.threads_per_block, "measured_iteration", &timing_stats, cfg.execution, cfg.batch_rows);
                    }
                }

                {
                    NvtxRange summary_range("timing_summary");
                    gq::print_pipeline_timing_summary(timing_stats, timing_mode);
                    gpu_metrics.total_gpu_ms = timing_stats.avg_total_gpu_ms();
                }

                if (cfg.validate) {
                    NvtxRange validate_range("cpu_validate");
                    const gq::GroupByPassengerFilteredResult cpu_gb =
                        gq::filter_groupby_passenger_count_soa(dataset, static_cast<std::size_t>(cfg.num_groups));
                    gpu_metrics.key_cardinality = count_nonempty_groupby_buckets(cpu_gb);
                    print_groupby_per_key_correctness(cpu_gb, gpu_gb);
                }
                return 0;
#endif
            }
        }

        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }

        std::string cmd = argv[1];

        if (cmd == "csv2bin") {
            if (argc < 4) {
                std::cerr << "Usage: " << argv[0] << " csv2bin <input.csv> <output.bin>\n";
                return 1;
            }
            std::string input_csv = argv[2];
            std::string output_bin = argv[3];

            gq::CsvReader reader;
            auto result = reader.read_soa(input_csv);
            gq::write_binary(output_bin, result.columns, result.rows);
            std::cout << "Converted " << input_csv << " to " << output_bin << " (" << result.rows << " rows)\n";
            return 0;
        } else if (cmd == "runbin_gpu_atomic") {
#ifndef GQUERY_USE_CUDA
            std::cerr << "error: CUDA not enabled in this build\n";
            return 1;
#else
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0]
                          << " runbin_gpu_atomic <input.bin> [iterations] [threads_per_block]\n";
                return 1;
            }
            std::string input_bin = argv[2];
            int iterations = (argc > 3) ? std::stoi(argv[3]) : 1;
            int threads_per_block = (argc > 4) ? std::stoi(argv[4]) : 256;

            gq::CpuTimer total_timer;
            gq::CpuTimer load_timer;
            auto bin_result = gq::read_binary(input_bin);
            double load_ms = load_timer.elapsed_ms();
            
            auto card = gq::compute_cardinality_soa(bin_result.columns);
            print_cardinality(card);

            gq::FilterGpuMetrics gpu_metrics{};
            const gq::Result gpu_result =
                gq::filter_and_sum_gpu_atomic_baseline(bin_result.columns, iterations, gpu_metrics, threads_per_block);
            double total_ms = total_timer.elapsed_ms();

            print_runbin_gpu_metrics(bin_result.file_bytes, bin_result.rows, iterations, load_ms, threads_per_block,
                                     gpu_metrics, total_ms, gpu_result.count, gpu_result.sum_fare_amount);
            const gq::Result cpu_result = gq::filter_and_sum_soa(bin_result.columns);
            print_cpu_gpu_correctness(cpu_result, gpu_result);
            return 0;
#endif
        } else if (cmd == "runbin_gpu_groupby_atomic") {
#ifndef GQUERY_USE_CUDA
            std::cerr << "error: CUDA not enabled in this build\n";
            return 1;
#else
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0]
                          << " runbin_gpu_groupby_atomic <input.bin> [iterations] [threads_per_block]\n";
                return 1;
            }
            std::string input_bin = argv[2];
            const int iterations = (argc > 3) ? std::stoi(argv[3]) : 1;
            const int threads_per_block = (argc > 4) ? std::stoi(argv[4]) : 256;

            gq::CpuTimer total_timer;
            gq::CpuTimer load_timer;
            auto bin_result = gq::read_binary(input_bin);
            const double load_ms = load_timer.elapsed_ms();

            auto card = gq::compute_cardinality_soa(bin_result.columns);
            print_cardinality(card);

            const std::size_t num_groups = static_cast<std::size_t>(card.max_key) + 1U;
            const gq::GroupByPassengerFilteredResult cpu_gb =
                gq::filter_groupby_passenger_count_soa(bin_result.columns, num_groups);

            gq::FilterGpuMetrics gpu_metrics{};
            const gq::GroupByGpuTable gpu_gb =
                gq::filter_groupby_gpu_atomic_baseline(bin_result.columns, iterations, gpu_metrics, num_groups,
                                                       threads_per_block);
            const double total_ms = total_timer.elapsed_ms();

            gpu_metrics.max_key_observed = static_cast<int>(card.max_key);
            gpu_metrics.key_cardinality = count_nonempty_groupby_buckets(cpu_gb);

            if (cpu_gb.out_of_range_selected_rows != 0U) {
                std::cout << "note: WHERE matched " << cpu_gb.out_of_range_selected_rows
                          << " row(s) with key outside [0," << num_groups << "); omitted from keyed aggregates.\n";
            }

            print_runbin_gpu_groupby_metrics(bin_result.file_bytes, bin_result.rows, iterations, load_ms,
                                             threads_per_block, gpu_metrics, total_ms, "gpu_groupby_atomic",
                                             "naive_atomic_per_key_global");
            print_groupby_per_key_correctness(cpu_gb, gpu_gb);
            return 0;
#endif
        } else if (cmd == "runbin_gpu_groupby_block_partial") {
#ifndef GQUERY_USE_CUDA
            std::cerr << "error: CUDA not enabled in this build\n";
            return 1;
#else
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0]
                          << " runbin_gpu_groupby_block_partial <input.bin> [iterations] [threads_per_block]\n";
                return 1;
            }
            std::string input_bin = argv[2];
            const int iterations = (argc > 3) ? std::stoi(argv[3]) : 1;
            const int threads_per_block = (argc > 4) ? std::stoi(argv[4]) : 256;

            gq::CpuTimer total_timer;
            gq::CpuTimer load_timer;
            auto bin_result = gq::read_binary(input_bin);
            const double load_ms = load_timer.elapsed_ms();

            auto card = gq::compute_cardinality_soa(bin_result.columns);
            print_cardinality(card);

            const std::size_t num_groups = static_cast<std::size_t>(card.max_key) + 1U;
            const gq::GroupByPassengerFilteredResult cpu_gb =
                gq::filter_groupby_passenger_count_soa(bin_result.columns, num_groups);

            gq::FilterGpuMetrics gpu_metrics{};
            const gq::GroupByGpuTable gpu_gb = gq::filter_groupby_gpu_compact_block_partial(
                bin_result.columns, iterations, gpu_metrics, num_groups, threads_per_block);
            const double total_ms = total_timer.elapsed_ms();

            gpu_metrics.max_key_observed = static_cast<int>(card.max_key);
            gpu_metrics.key_cardinality = count_nonempty_groupby_buckets(cpu_gb);

            if (cpu_gb.out_of_range_selected_rows != 0U) {
                std::cout << "note: WHERE matched " << cpu_gb.out_of_range_selected_rows
                          << " row(s) with key outside [0," << num_groups << "); omitted from keyed aggregates.\n";
            }

            print_runbin_gpu_groupby_metrics(bin_result.file_bytes, bin_result.rows, iterations, load_ms,
                                             threads_per_block, gpu_metrics, total_ms, "gpu_groupby_block_partial",
                                             "compact_block_partial_per_key_merge");
            print_groupby_per_key_correctness(cpu_gb, gpu_gb);
            return 0;
#endif
        } else if (cmd == "runbin_gpu_mask" || cmd == "runbin_gpu_mask_atomic" || cmd == "runbin_gpu_mask_block_partial") {
#ifndef GQUERY_USE_CUDA
            std::cerr << "error: CUDA not enabled in this build\n";
            return 1;
#else
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0]
                          << " runbin_gpu_mask[_atomic|_block_partial] <input.bin> [iterations] [threads_per_block]\n";
                return 1;
            }
            std::string input_bin = argv[2];
            int iterations = (argc > 3) ? std::stoi(argv[3]) : 1;
            int threads_per_block = (argc > 4) ? std::stoi(argv[4]) : 256;

            gq::CpuTimer total_timer;
            gq::CpuTimer load_timer;
            auto bin_result = gq::read_binary(input_bin);
            double load_ms = load_timer.elapsed_ms();
            
            auto card = gq::compute_cardinality_soa(bin_result.columns);
            print_cardinality(card);

            gq::FilterGpuMetrics gpu_metrics{};
            gq::Result gpu_result{};
            std::string reduce_algorithm = "atomic";
            if (cmd == "runbin_gpu_mask_block_partial") {
                gpu_result =
                    gq::filter_and_sum_gpu_compact_block_partial(bin_result.columns, iterations, gpu_metrics, threads_per_block);
                reduce_algorithm = "block_partial";
            } else {
                gpu_result = gq::filter_and_sum_gpu_compact_atomic(bin_result.columns, iterations, gpu_metrics,
                                                                   threads_per_block);
            }
            double total_ms = total_timer.elapsed_ms();

            print_runbin_gpu_mask_metrics(bin_result.file_bytes, bin_result.rows, iterations, load_ms, threads_per_block,
                                          gpu_metrics, total_ms, gpu_result.count, gpu_result.sum_fare_amount,
                                          reduce_algorithm);
            const gq::Result cpu_result = gq::filter_and_sum_soa(bin_result.columns);
            print_cpu_gpu_correctness(cpu_result, gpu_result);
            return 0;
#endif
        } else if (cmd == "runbin") {
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0] << " runbin <input.bin> [iterations]\n";
                return 1;
            }
            std::string input_bin = argv[2];
            int iterations = (argc > 3) ? std::stoi(argv[3]) : 1;

            gq::CpuTimer total_timer;
            // Load binary
            gq::CpuTimer load_timer;
            auto bin_result = gq::read_binary(input_bin);
            double load_ms = load_timer.elapsed_ms();
            
            auto card = gq::compute_cardinality_soa(bin_result.columns);
            print_cardinality(card);

            gq::Result filter_result{};
#ifdef GQUERY_USE_CUDA
            gq::FilterGpuMetrics gpu_metrics{};
            filter_result = gq::filter_and_sum_gpu_atomic_baseline(bin_result.columns, iterations, gpu_metrics);
            double total_ms = total_timer.elapsed_ms();
            print_runbin_gpu_metrics(bin_result.file_bytes, bin_result.rows, iterations, load_ms, /*threads_per_block=*/256,
                                     gpu_metrics, total_ms, filter_result.count, filter_result.sum_fare_amount);
            // Correctness check against CPU on the same input. Count must match exactly; sum may differ slightly due to
            // floating-point atomic accumulation order on GPU.
            const gq::Result cpu_result = gq::filter_and_sum_soa(bin_result.columns);
            print_cpu_gpu_correctness(cpu_result, filter_result);
#else
            gq::CpuTimer filter_timer;
            for (int i = 0; i < iterations; ++i) {
                filter_result = gq::filter_and_sum_soa(bin_result.columns);
            }
            double total_filter_ms = filter_timer.elapsed_ms();
            double avg_filter_ms = total_filter_ms / static_cast<double>(iterations);
            double total_ms = total_timer.elapsed_ms();
            print_metrics("runbin", "soa", bin_result.file_bytes, bin_result.rows, iterations, load_ms, avg_filter_ms,
                          total_ms, filter_result.count, filter_result.sum_fare_amount, 0.0);
#endif
            return 0;
        } else {
            // Default to CSV mode for backward compatibility with previous task
            std::string input_csv = cmd;
            int iterations = (argc > 2) ? std::stoi(argv[2]) : 1;

            gq::CpuTimer total_timer;

            // 1. Parsing once
            gq::CpuTimer parse_timer;
            gq::CsvReader reader;
            auto read_result = reader.read_soa(input_csv);
            double parse_ms = parse_timer.elapsed_ms();

            auto card = gq::compute_cardinality_soa(read_result.columns);
            print_cardinality(card);

            // 2. Filtering repeatedly
            gq::Result filter_result{};
            gq::CpuTimer filter_timer;
            for (int i = 0; i < iterations; ++i) {
                filter_result = gq::filter_and_sum_soa(read_result.columns);
            }
            double total_filter_ms = filter_timer.elapsed_ms();
            double avg_filter_ms = total_filter_ms / static_cast<double>(iterations);

            double total_ms = total_timer.elapsed_ms();
            
            double parse_seconds = parse_ms / 1000.0;
            double parse_mb_per_s = parse_seconds > 0.0 ? (static_cast<double>(read_result.file_bytes) / parse_seconds / 1e6) : 0.0;

            print_metrics("cpu", "soa", read_result.file_bytes, read_result.rows, iterations, parse_ms, avg_filter_ms, total_ms, filter_result.count, filter_result.sum_fare_amount, parse_mb_per_s);
            return 0;
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
