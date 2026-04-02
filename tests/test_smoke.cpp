#include "gq/filter_cpu.hpp"
#include "gq/csv_reader.hpp"

#include <cassert>
#include <iostream>
#include <filesystem>

void test_filter() {
    gq::Columns columns{};
    columns.trip_distance = {1.0f, 3.0f, 4.0f};
    columns.fare_amount   = {20.0f, 5.0f, 12.0f};

    const gq::Result result = gq::filter_and_sum_soa(columns);

    assert(result.count == 1);
    assert(result.sum_fare_amount == 12.0);
}

void test_csv_reader() {
    gq::CsvReader reader;
    // Try both local and parent directory paths to handle different build environments
    std::string path = "data/test_5_rows.csv";
    if (!std::filesystem::exists(path)) {
        path = "../data/test_5_rows.csv";
    }
    
    auto result = reader.read_soa(path);
    
    assert(result.rows == 5);
    assert(result.columns.trip_distance.size() == 5);
    assert(result.columns.fare_amount.size() == 5);
    
    assert(result.columns.trip_distance[0] == 1.5f);
    assert(result.columns.fare_amount[0] == 10.0f);
    
    assert(result.columns.trip_distance[4] == 5.5f);
    assert(result.columns.fare_amount[4] == 50.0f);
    
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

    assert(filter_result.count == 2);
    assert(filter_result.sum_fare_amount == 23.5);

    std::cout << "Sample CSV test passed!\n";
}

int main() {
    try {
        test_filter();
        test_csv_reader();
        test_sample_csv();
        std::cout << "Smoke test passed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
