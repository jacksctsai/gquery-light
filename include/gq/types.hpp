#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

#ifdef GQUERY_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace gq {

// GROUP BY SUM(fare_amount) keyed by passenger_count: supported keys are
// integers in [0, kMaxPassengerCountKey). Keys outside this range are skipped on GPU / tracked on CPU reference.
inline constexpr int kMaxPassengerCountKey = 9;

struct Row {
    float trip_distance{};
    float fare_amount{};
};

#ifdef GQUERY_USE_CUDA
template <class T>
class CudaPinnedAllocator {
public:
    using value_type = T;

    CudaPinnedAllocator() noexcept = default;
    template <class U>
    CudaPinnedAllocator(const CudaPinnedAllocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        void* p = nullptr;
        const cudaError_t err = cudaMallocHost(&p, n * sizeof(T));
        if (err != cudaSuccess || p == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept {
        if (p) {
            cudaFreeHost(p);
        }
    }
};

template <class T, class U>
inline bool operator==(const CudaPinnedAllocator<T>&, const CudaPinnedAllocator<U>&) noexcept {
    return true;
}
template <class T, class U>
inline bool operator!=(const CudaPinnedAllocator<T>&, const CudaPinnedAllocator<U>&) noexcept {
    return false;
}

using HostFloatVector = std::vector<float, CudaPinnedAllocator<float>>;
using HostUint8Vector = std::vector<std::uint8_t, CudaPinnedAllocator<std::uint8_t>>;
#else
using HostFloatVector = std::vector<float>;
using HostUint8Vector = std::vector<std::uint8_t>;
#endif

struct Columns {
    HostUint8Vector passenger_count;
    HostFloatVector trip_distance;
    HostFloatVector fare_amount;
};

struct Result {
    std::uint64_t count{0};
    double sum_fare_amount{0.0};
};

struct GroupByPassengerFilteredResult {
    std::array<std::uint64_t, static_cast<std::size_t>(kMaxPassengerCountKey)> count{};
    std::array<double, static_cast<std::size_t>(kMaxPassengerCountKey)> sum{};
    std::uint64_t selected_matching_rows{};
    /** Matching predicate but passenger_count outside [0, kMaxPassengerCountKey). */
    std::uint64_t out_of_range_selected_rows{};
};

struct GroupByGpuTable {
    std::array<std::uint64_t, static_cast<std::size_t>(kMaxPassengerCountKey)> count{};
    std::array<double, static_cast<std::size_t>(kMaxPassengerCountKey)> sum{};
};

} // namespace gq