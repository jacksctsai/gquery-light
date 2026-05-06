#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>

#include "gq/types.hpp"

namespace gq {

struct CsvReadResult {
    Columns columns;
    std::uint64_t rows{0};
    std::uint64_t file_bytes{0};
};

struct CsvLayout {
    std::size_t passenger_count_idx{};
    std::size_t trip_distance_idx{};
    std::size_t fare_amount_idx{};
};

class CsvReader {
public:
    CsvReadResult read_soa(const std::filesystem::path& path) const;

private:
    static std::vector<std::string> split_header(const std::string& header);
    static std::size_t find_column_index(const std::vector<std::string>& columns, const std::string& column_name);
    static CsvLayout resolve_layout(const std::string& header);
    
    // Pass 1: Row counting
    static std::uint64_t count_data_rows(std::ifstream& in);
    
    // Pass 2: Field extraction
    static void extract_target_fields(
        const std::string& line,
        const CsvLayout& layout,
        std::uint8_t& passenger_count,
        float& trip_distance,
        float& fare_amount,
        std::uint64_t line_number);

    static float parse_float_token(const std::string& line, std::size_t start, std::size_t end, const char* column_name, std::uint64_t line_number);
    static std::uint8_t parse_uint8_token(const std::string& line, std::size_t start, std::size_t end, const char* column_name, std::uint64_t line_number);
};

} // namespace gq
