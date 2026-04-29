#include <exception>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>

#include "gq/csv_reader.hpp"
#include "gq/filter_cpu.hpp"
#include "gq/timer.hpp"
#include "gq/binary_io.hpp"

#ifdef GQUERY_USE_CUDA
#include "gq/filter_gpu.hpp"
#endif

void print_metrics(const std::string& mode, const std::string& layout, std::uint64_t file_bytes, std::uint64_t rows, int iterations, double load_ms, double filter_ms_avg, double total_ms, std::uint64_t count, double sum_fare_amount, double parse_mb_per_s) {
    
    std::uint64_t payload_bytes = rows * 2 * sizeof(float);
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

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: \n"
                      << "  " << argv[0] << " csv2bin <input.csv> <output.bin>\n"
                      << "  " << argv[0] << " runbin <input.bin> <iterations>\n"
                      << "  " << argv[0] << " runbin_gpu_atomic <input.bin> [iterations] [threads_per_block]\n"
                      << "  " << argv[0] << " runbin_gpu_mask_atomic <input.bin> [iterations] [threads_per_block]\n"
                      << "  " << argv[0] << " runbin_gpu_mask_block_partial <input.bin> [iterations] [threads_per_block]\n"
                      << "  " << argv[0] << " runbin_gpu_mask <input.bin> [iterations] [threads_per_block]\n"
                      << "  " << argv[0] << " <input.csv> [iterations]\n";
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
