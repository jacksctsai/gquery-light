#pragma once

#include <cstdint>
#include <new>
#include <vector>

#ifdef GQUERY_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace gq {

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
#else
using HostFloatVector = std::vector<float>;
#endif

struct Columns {
    HostFloatVector trip_distance;
    HostFloatVector fare_amount;
};

struct Result {
    std::uint64_t count{0};
    double sum_fare_amount{0.0};
};

} // namespace gq