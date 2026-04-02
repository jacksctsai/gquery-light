#include "gq/binary_io.hpp"
#include <fstream>
#include <stdexcept>

namespace gq {

void write_binary(const std::filesystem::path& path, const Columns& columns, std::uint64_t rows) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open file for writing: " + path.string());
    }
    
    // Header: uint64_t row_count;
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    
    // Then raw arrays:
    // float trip_distance[row_count];
    out.write(reinterpret_cast<const char*>(columns.trip_distance.data()), static_cast<std::streamsize>(rows * sizeof(float)));
    
    // float fare_amount[row_count];
    out.write(reinterpret_cast<const char*>(columns.fare_amount.data()), static_cast<std::streamsize>(rows * sizeof(float)));
}

BinaryReadResult read_binary(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open file for reading: " + path.string());
    }
    
    std::uint64_t rows = 0;
    if (!in.read(reinterpret_cast<char*>(&rows), sizeof(rows))) {
        throw std::runtime_error("Could not read row count from binary file: " + path.string());
    }
    
    Columns columns;
    columns.trip_distance.resize(rows);
    columns.fare_amount.resize(rows);
    
    if (!in.read(reinterpret_cast<char*>(columns.trip_distance.data()), static_cast<std::streamsize>(rows * sizeof(float)))) {
        throw std::runtime_error("Could not read trip_distance array from binary file: " + path.string());
    }
    
    if (!in.read(reinterpret_cast<char*>(columns.fare_amount.data()), static_cast<std::streamsize>(rows * sizeof(float)))) {
        throw std::runtime_error("Could not read fare_amount array from binary file: " + path.string());
    }
    
    std::uint64_t file_bytes = std::filesystem::file_size(path);
    
    return {std::move(columns), rows, file_bytes};
}

} // namespace gq
