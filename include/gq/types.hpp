#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef GQUERY_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace gq {

// NOTE: Month 3 benchmark makes GROUP BY cardinality configurable at runtime.

struct Row {
    float trip_distance{};
    float fare_amount{};
};

enum class HostMemoryMode {
    Pageable,
    Pinned
};

enum class ExecutionMode {
    Sync,
    SingleStreamAsync,
    MultiStreamBatched
};

#ifdef GQUERY_USE_CUDA
inline void cuda_host_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}
#endif

template <class T>
class HostVector {
public:
    HostVector() noexcept = default;

    HostVector(std::initializer_list<T> init) {
        assign(init);
    }

    HostVector(const HostVector&) = delete;
    HostVector& operator=(const HostVector&) = delete;

    HostVector(HostVector&& other) noexcept
        : data_(other.data_), size_(other.size_), pinned_(other.pinned_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.pinned_ = false;
    }

    HostVector& operator=(HostVector&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            size_ = other.size_;
            pinned_ = other.pinned_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.pinned_ = false;
        }
        return *this;
    }

    HostVector& operator=(std::initializer_list<T> init) {
        assign(init);
        return *this;
    }

    ~HostVector() {
        release();
    }

    void resize(std::size_t n, bool pinned = false) {
        if (n == size_ && pinned == pinned_) {
            return;
        }
        release();
        size_ = n;
        pinned_ = pinned;
        if (n == 0) {
            return;
        }
#ifdef GQUERY_USE_CUDA
        if (pinned_) {
            cuda_host_check(cudaMallocHost(reinterpret_cast<void**>(&data_), n * sizeof(T)), "cudaMallocHost");
        } else {
            data_ = new T[n];
        }
#else
        if (pinned_) {
            throw std::runtime_error("pinned host memory requires CUDA build");
        }
        data_ = new T[n];
#endif
    }

    [[nodiscard]] T& operator[](std::size_t i) {
        return data_[i];
    }
    [[nodiscard]] const T& operator[](std::size_t i) const {
        return data_[i];
    }

    [[nodiscard]] T* data() noexcept {
        return data_;
    }
    [[nodiscard]] const T* data() const noexcept {
        return data_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] bool is_pinned() const noexcept {
        return pinned_;
    }

    [[nodiscard]] T* begin() noexcept {
        return data_;
    }
    [[nodiscard]] const T* begin() const noexcept {
        return data_;
    }
    [[nodiscard]] T* end() noexcept {
        return data_ + size_;
    }
    [[nodiscard]] const T* end() const noexcept {
        return data_ + size_;
    }

private:
    void assign(std::initializer_list<T> init) {
        resize(init.size(), false);
        std::size_t i = 0;
        for (const T& v : init) {
            data_[i++] = v;
        }
    }

    void release() noexcept {
        if (data_ == nullptr) {
            size_ = 0;
            pinned_ = false;
            return;
        }
#ifdef GQUERY_USE_CUDA
        if (pinned_) {
            cudaFreeHost(data_);
        } else {
            delete[] data_;
        }
#else
        delete[] data_;
#endif
        data_ = nullptr;
        size_ = 0;
        pinned_ = false;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
    bool pinned_ = false;
};

using HostFloatVector = HostVector<float>;
using HostUint8Vector = HostVector<std::uint8_t>;
using HostUint32Vector = HostVector<std::uint32_t>;

struct Columns {
    HostUint32Vector passenger_count;
    HostFloatVector trip_distance;
    HostFloatVector fare_amount;
};

struct Result {
    std::uint64_t count{0};
    double sum_fare_amount{0.0};
};

struct GroupByPassengerFilteredResult {
    std::vector<std::uint64_t> count;
    std::vector<double> sum;
    std::uint64_t selected_matching_rows{};
    /** Matching predicate but passenger_count outside [0, num_groups). */
    std::uint64_t out_of_range_selected_rows{};
};

struct GroupByGpuTable {
    std::vector<std::uint64_t> count;
    std::vector<double> sum;
};

} // namespace gq
