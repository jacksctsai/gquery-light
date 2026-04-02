#pragma once

#include "gq/types.hpp"
#include <cstdint>

namespace gq {

/**
 * @brief Run the filter on the GPU.
 * 
 * @param columns Input data in SoA format.
 * @param iterations Number of times to run the filter (for benchmark).
 * @param filter_ms Output: average time taken for filtering (ms).
 * @return Result 
 */
Result filter_and_sum_gpu(const Columns& columns, int iterations, double& filter_ms);

} // namespace gq
