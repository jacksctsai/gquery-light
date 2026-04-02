#include "gq/filter_cpu.hpp"

#include <stdexcept>

namespace gq {

Result filter_and_sum_soa(const Columns& columns) {
    if (columns.trip_distance.size() != columns.fare_amount.size()) {
        throw std::runtime_error("column size mismatch");
    }

    Result result{};

    const std::size_t n = columns.trip_distance.size();
    for (std::size_t i = 0; i < n; ++i) {
        const float trip_distance = columns.trip_distance[i];
        const float fare_amount = columns.fare_amount[i];

        if (trip_distance > 2.5f && fare_amount > 10.0f) {
            ++result.count;
            result.sum_fare_amount += static_cast<double>(fare_amount);
        }
    }

    return result;
}

} // namespace gq