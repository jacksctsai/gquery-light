#include "gq/filter_cpu.hpp"

#include <stdexcept>
#include <algorithm>
#include <limits>

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

GroupByPassengerFilteredResult filter_groupby_passenger_count_soa(const Columns& columns, std::size_t num_groups) {
    if (columns.trip_distance.size() != columns.fare_amount.size()) {
        throw std::runtime_error("column size mismatch");
    }
    if (columns.passenger_count.size() != columns.trip_distance.size()) {
        throw std::runtime_error("passenger_count column size mismatch");
    }
    if (num_groups == 0) {
        throw std::runtime_error("num_groups must be > 0");
    }

    GroupByPassengerFilteredResult out{};
    out.count.assign(num_groups, 0U);
    out.sum.assign(num_groups, 0.0);
    const std::size_t n = columns.trip_distance.size();
    for (std::size_t i = 0; i < n; ++i) {
        const float trip_distance = columns.trip_distance[i];
        const float fare_amount = columns.fare_amount[i];
        if (trip_distance > 2.5f && fare_amount > 10.0f) {
            ++out.selected_matching_rows;
            const std::uint32_t pk = columns.passenger_count[i];
            if (pk < static_cast<std::uint32_t>(num_groups)) {
                const std::size_t b = static_cast<std::size_t>(pk);
                ++out.count[b];
                out.sum[b] += static_cast<double>(fare_amount);
            } else {
                ++out.out_of_range_selected_rows;
            }
        }
    }
    return out;
}

CardinalityResult compute_cardinality_soa(const Columns& columns) {
    CardinalityResult res;
    if (columns.passenger_count.empty()) {
         return res;
    }
    
    for (const auto& pc : columns.passenger_count) {
        res.counts[pc]++;
        res.min_key = std::min(res.min_key, pc);
        res.max_key = std::max(res.max_key, pc);
    }
    return res;
}

} // namespace gq