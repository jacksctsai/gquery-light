#include "gq/binary_io.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace gq {

const char MAGIC_V2[4] = {'G', 'Q', '0', '2'};

void write_binary(const std::filesystem::path& path, const Columns& columns, std::uint64_t rows) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open file for writing: " + path.string());
    }
    
    // Write V2 Magic Header
    out.write(MAGIC_V2, sizeof(MAGIC_V2));

    // Header: uint64_t row_count;
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    
    // Then raw arrays:
    // uint8_t passenger_count[row_count];
    out.write(reinterpret_cast<const char*>(columns.passenger_count.data()), static_cast<std::streamsize>(rows * sizeof(std::uint8_t)));

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
    
    char magic[4] = {0};
    in.read(magic, sizeof(magic));
    
    bool is_v2 = (std::memcmp(magic, MAGIC_V2, sizeof(magic)) == 0);
    
    std::uint64_t rows = 0;
    if (is_v2) {
        if (!in.read(reinterpret_cast<char*>(&rows), sizeof(rows))) {
            throw std::runtime_error("Could not read row count from V2 binary file: " + path.string());
        }
    } else {
        // Fallback to V1 where the first 8 bytes were the row count
        std::memcpy(&rows, magic, sizeof(magic)); // Copy the first 4 bytes we accidentally read
        char next4[4];
        if (!in.read(next4, sizeof(next4))) {
             throw std::runtime_error("Could not read row count from V1 binary file: " + path.string());
        }
        std::memcpy(reinterpret_cast<char*>(&rows) + sizeof(magic), next4, sizeof(next4));
        std::cerr << "Warning: Loading legacy V1 binary format without passenger_count.\n";
    }

    Columns columns;
    columns.passenger_count.resize(rows, 0); // V1 will have 0s
    columns.trip_distance.resize(rows);
    columns.fare_amount.resize(rows);
    
    if (is_v2) {
        if (!in.read(reinterpret_cast<char*>(columns.passenger_count.data()), static_cast<std::streamsize>(rows * sizeof(std::uint8_t)))) {
            throw std::runtime_error("Could not read passenger_count array from binary file: " + path.string());
        }
    }

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
