#include "gq/filter_cpu.hpp"
#include "gq/csv_reader.hpp"

#include <cassert>
#include <iostream>
#include <filesystem>

void test_filter() {
    gq::Columns columns{};
    columns.trip_distance = {1.0f, 3.0f, 4.0f};
    columns.fare_amount   = {20.0f, 5.0f, 12.0f};
    columns.passenger_count.resize(columns.trip_distance.size(), 0);

    const gq::Result result = gq::filter_and_sum_soa(columns);

    if (result.count != 1) throw std::runtime_error("test_filter: count mismatch");
    if (result.sum_fare_amount != 12.0) throw std::runtime_error("test_filter: sum mismatch");
}

void test_groupby_passenger_count_cpu() {
    gq::Columns columns{};
    columns.passenger_count = {1, 2, 8, 9};
    columns.trip_distance = {3.0f, 3.0f, 3.0f, 3.0f};
    columns.fare_amount = {15.0f, 20.0f, 12.0f, 30.0f};

    const std::size_t num_groups = 9; // keys [0..8]
    const gq::GroupByPassengerFilteredResult gb = gq::filter_groupby_passenger_count_soa(columns, num_groups);
    if (gb.selected_matching_rows != 4) throw std::runtime_error("test_groupby: selected_matching_rows mismatch");
    if (gb.out_of_range_selected_rows != 1) throw std::runtime_error("test_groupby: out_of_range mismatch"); // key==9
    if (gb.count.size() != num_groups || gb.sum.size() != num_groups) throw std::runtime_error("test_groupby: shape mismatch");
    if (gb.count[1] != 1 || gb.sum[1] != 15.0) throw std::runtime_error("test_groupby: bucket 1 mismatch");
    if (gb.count[2] != 1 || gb.sum[2] != 20.0) throw std::runtime_error("test_groupby: bucket 2 mismatch");
    if (gb.count[8] != 1 || std::abs(gb.sum[8] - 12.0) >= 1e-9) throw std::runtime_error("test_groupby: bucket 8 mismatch");

    bool others_zero = true;
    for (std::size_t k = 0; k < num_groups; ++k) {
        if (k == 1 || k == 2 || k == 8) {
            continue;
        }
        if (gb.count[k] != 0) {
            others_zero = false;
        }
    }
    if (!others_zero) throw std::runtime_error("test_groupby: unexpected nonzero bucket");
}

void test_csv_reader() {
    gq::CsvReader reader;
    // Try both local and parent directory paths to handle different build environments
    std::string path = "data/test_5_rows.csv";
    if (!std::filesystem::exists(path)) {
        path = "../data/test_5_rows.csv";
    }
    
    auto result = reader.read_soa(path);
    
    if (result.rows != 5) throw std::runtime_error("test_csv_reader: rows mismatch");
    if (result.columns.trip_distance.size() != 5) throw std::runtime_error("test_csv_reader: trip_distance size mismatch");
    if (result.columns.fare_amount.size() != 5) throw std::runtime_error("test_csv_reader: fare_amount size mismatch");
    if (result.columns.trip_distance[0] != 1.5f) throw std::runtime_error("test_csv_reader: trip_distance[0] mismatch");
    if (result.columns.fare_amount[0] != 10.0f) throw std::runtime_error("test_csv_reader: fare_amount[0] mismatch");
    if (result.columns.trip_distance[4] != 5.5f) throw std::runtime_error("test_csv_reader: trip_distance[4] mismatch");
    if (result.columns.fare_amount[4] != 50.0f) throw std::runtime_error("test_csv_reader: fare_amount[4] mismatch");
    
    std::cout << "CSV reader test passed!\n";
}

void test_sample_csv() {
    gq::CsvReader reader;
    std::string path = "data/sample.csv";
    if (!std::filesystem::exists(path)) {
        path = "../data/sample.csv";
    }

    if (!std::filesystem::exists(path)) {
        std::cout << "Skipping test_sample_csv: file not found.\n";
        return;
    }

    auto result = reader.read_soa(path);
    auto filter_result = gq::filter_and_sum_soa(result.columns);

    if (filter_result.count != 2) throw std::runtime_error("test_sample_csv: count mismatch");
    if (filter_result.sum_fare_amount != 23.5) throw std::runtime_error("test_sample_csv: sum mismatch");

    std::cout << "Sample CSV test passed!\n";
}

int main() {
    try {
        test_filter();
        test_groupby_passenger_count_cpu();
        test_csv_reader();
        test_sample_csv();
        std::cout << "Smoke test passed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
