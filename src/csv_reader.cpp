#include "gq/csv_reader.hpp"

#include <stdexcept>
#include <sstream>
#include <fstream>
#include <string>

static void trim_cr(std::string& s) {
    if (!s.empty() && s.back() == '\r') {
        s.pop_back();
    }
}

namespace gq {

std::vector<std::string> CsvReader::split_header(const std::string& header) {
    std::vector<std::string> columns;
    std::size_t start = 0;
    std::size_t end = 0;
    while ((end = header.find(',', start)) != std::string::npos) {
        columns.push_back(header.substr(start, end - start));
        start = end + 1;
    }
    columns.push_back(header.substr(start));
    return columns;
}

std::size_t CsvReader::find_column_index(const std::vector<std::string>& columns, const std::string& column_name) {
    for (std::size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == column_name) {
            return i;
        }
    }
    throw std::runtime_error("Column not found: " + column_name);
}

CsvLayout CsvReader::resolve_layout(const std::string& header) {
    auto columns = split_header(header);
    CsvLayout layout;
    layout.passenger_count_idx = find_column_index(columns, "passenger_count");
    layout.trip_distance_idx = find_column_index(columns, "trip_distance");
    layout.fare_amount_idx = find_column_index(columns, "fare_amount");
    return layout;
}

std::uint64_t CsvReader::count_data_rows(std::ifstream& in) {
    std::uint64_t count = 0;
    std::string line;
    // Skip header line
    if (!std::getline(in, line)) {
        return 0;
    }
    while (std::getline(in, line)) {
        if (!line.empty()) {
            count++;
        }
    }
    return count;
}

float CsvReader::parse_float_token(const std::string& line, std::size_t start, std::size_t end, const char* column_name, std::uint64_t line_number) {
    if (start >= end) {
        throw std::runtime_error("Empty token for column " + std::string(column_name) + " at line " + std::to_string(line_number));
    }
    try {
        return std::stof(line.substr(start, end - start));
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse " + std::string(column_name) + " at line " + std::to_string(line_number) + ": " + line.substr(start, end - start));
    }
}

std::uint8_t CsvReader::parse_uint8_token(const std::string& line, std::size_t start, std::size_t end, const char* column_name, std::uint64_t line_number) {
    if (start >= end) {
        // Defensively handle missing/empty passenger counts
        return 0;
    }
    try {
        int val = std::stoi(line.substr(start, end - start));
        if (val < 0 || val > 255) {
             return 0; // Out of bounds, defensive fallback
        }
        return static_cast<std::uint8_t>(val);
    } catch (const std::exception& e) {
        return 0; // Malformed data, defensive fallback
    }
}

void CsvReader::extract_target_fields(
    const std::string& line,
    const CsvLayout& layout,
    std::uint8_t& passenger_count,
    float& trip_distance,
    float& fare_amount,
    std::uint64_t line_number) {
    
    std::size_t current_col = 0;
    std::size_t start = 0;
    std::size_t end = 0;
    
    bool passenger_count_found = false;
    bool trip_distance_found = false;
    bool fare_amount_found = false;

    while ((end = line.find(',', start)) != std::string::npos) {
        if (current_col == layout.passenger_count_idx) {
            passenger_count = parse_uint8_token(line, start, end, "passenger_count", line_number);
            passenger_count_found = true;
        } else if (current_col == layout.trip_distance_idx) {
            trip_distance = parse_float_token(line, start, end, "trip_distance", line_number);
            trip_distance_found = true;
        } else if (current_col == layout.fare_amount_idx) {
            fare_amount = parse_float_token(line, start, end, "fare_amount", line_number);
            fare_amount_found = true;
        }
        start = end + 1;
        current_col++;
    }
    
    // Check last column
    if (current_col == layout.passenger_count_idx) {
        passenger_count = parse_uint8_token(line, start, line.size(), "passenger_count", line_number);
        passenger_count_found = true;
    } else if (current_col == layout.trip_distance_idx) {
        trip_distance = parse_float_token(line, start, line.size(), "trip_distance", line_number);
        trip_distance_found = true;
    } else if (current_col == layout.fare_amount_idx) {
        fare_amount = parse_float_token(line, start, line.size(), "fare_amount", line_number);
        fare_amount_found = true;
    }

    if (!trip_distance_found || !fare_amount_found || !passenger_count_found) {
        throw std::runtime_error("Missing required columns in CSV at line " + std::to_string(line_number));
    }
}

CsvReadResult CsvReader::read_soa(const std::filesystem::path& path) const {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    // Pass 1: Count rows
    std::uint64_t rows = count_data_rows(in);
    
    // Reset to beginning for Pass 2
    in.clear();
    in.seekg(0, std::ios::beg);
    
    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("Empty file: " + path.string());
    }
    trim_cr(header);
    CsvLayout layout = resolve_layout(header);
    
    Columns columns;
    columns.passenger_count.resize(rows);
    columns.trip_distance.resize(rows);
    columns.fare_amount.resize(rows);
    
    std::string line;
    std::uint64_t current_row = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (current_row >= rows) {
            throw std::runtime_error("Row count mismatch");
        } // Should not happen if file hasn't changed
        trim_cr(line);
        extract_target_fields(line, layout, columns.passenger_count[current_row], columns.trip_distance[current_row], columns.fare_amount[current_row], current_row + 2);
        current_row++;
    }
    if (current_row != rows) {
        throw std::runtime_error("Row count mismatch at end");
    }
    std::uintmax_t file_bytes = std::filesystem::file_size(path);
    
    return {std::move(columns), rows, static_cast<std::uint64_t>(file_bytes)};
}

} // namespace gq
