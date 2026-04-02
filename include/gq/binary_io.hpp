#pragma once

#include <filesystem>
#include <cstdint>
#include <vector>
#include "gq/types.hpp"

namespace gq {

struct BinaryReadResult {
    Columns columns;
    std::uint64_t rows{0};
    std::uint64_t file_bytes{0};
};

void write_binary(const std::filesystem::path& path, const Columns& columns, std::uint64_t rows);
BinaryReadResult read_binary(const std::filesystem::path& path);

} // namespace gq
