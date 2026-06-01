#pragma once

#include "gq/types.hpp"
#include <map>
#include <limits>

namespace gq {

struct CardinalityResult {
    std::uint32_t min_key{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t max_key{0};
    std::map<std::uint32_t, std::uint64_t> counts;
};

Result filter_and_sum_soa(const Columns& columns);
CardinalityResult compute_cardinality_soa(const Columns& columns);

/** CPU reference for masked GROUP BY SUM(fare) GROUP BY passenger_count with same predicate as filter_and_sum_soa. */
GroupByPassengerFilteredResult filter_groupby_passenger_count_soa(const Columns& columns, std::size_t num_groups);

} // namespace gq