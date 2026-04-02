#include <exception>
#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>

#include "gq/csv_reader.hpp"
#include "gq/filter_cpu.hpp"
#include "gq/timer.hpp"
#include "gq/binary_io.hpp"

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

    // Machine-readable line
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

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: \n"
                      << "  " << argv[0] << " csv2bin <input.csv> <output.bin>\n"
                      << "  " << argv[0] << " runbin <input.bin> <iterations>\n"
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

            // Filter
            gq::Result filter_result{};
            gq::CpuTimer filter_timer;
            for (int i = 0; i < iterations; ++i) {
                filter_result = gq::filter_and_sum_soa(bin_result.columns);
            }
            double total_filter_ms = filter_timer.elapsed_ms();
            double avg_filter_ms = total_filter_ms / static_cast<double>(iterations);
            
            double total_ms = total_timer.elapsed_ms();

            print_metrics("runbin", "soa", bin_result.file_bytes, bin_result.rows, iterations, load_ms, avg_filter_ms, total_ms, filter_result.count, filter_result.sum_fare_amount, 0.0);
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
