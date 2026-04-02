#pragma once

#include <cstdint>
#include <vector>

namespace gq {

struct Row {
    float trip_distance{};
    float fare_amount{};
};

struct Columns {
    std::vector<float> trip_distance;
    std::vector<float> fare_amount;
};

struct Result {
    std::uint64_t count{0};
    double sum_fare_amount{0.0};
};

} // namespace gq