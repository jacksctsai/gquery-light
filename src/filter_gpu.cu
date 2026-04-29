#include "gq/filter_gpu.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gq {

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

inline void events_create(cudaEvent_t& a, cudaEvent_t& b, cudaEvent_t& c, cudaEvent_t& d) {
    cuda_check(cudaEventCreate(&a), "cudaEventCreate");
    cuda_check(cudaEventCreate(&b), "cudaEventCreate");
    cuda_check(cudaEventCreate(&c), "cudaEventCreate");
    cuda_check(cudaEventCreate(&d), "cudaEventCreate");
}

inline void events_destroy(cudaEvent_t a, cudaEvent_t b, cudaEvent_t c, cudaEvent_t d) {
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaEventDestroy(c);
    cudaEventDestroy(d);
}

inline double elapsed_ms(cudaEvent_t start, cudaEvent_t end, const char* what) {
    float ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&ms, start, end), what);
    return static_cast<double>(ms);
}

__global__ void filter_rows_atomic_kernel(const float* trip_distance, const float* fare_amount, std::size_t n,
                                          unsigned long long* count, double* sum_fare) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = tid; i < n; i += stride) {
        const float td = trip_distance[i];
        const float fa = fare_amount[i];
        if (td > 2.5f && fa > 10.0f) {
            atomicAdd(count, 1ULL);
            atomicAdd(sum_fare, static_cast<double>(fa));
        }
    }
}

__global__ void filter_mask_kernel(const float* trip_distance, const float* fare_amount, std::size_t n,
                                   std::uint8_t* mask) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = tid; i < n; i += stride) {
        const float td = trip_distance[i];
        const float fa = fare_amount[i];
        mask[i] = (td > 2.5f && fa > 10.0f) ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    }
}

} // namespace

Result filter_and_sum_gpu_atomic_baseline(const Columns& columns, int iterations, FilterGpuMetrics& metrics,
                                          int threads_per_block) {
    if (iterations < 1) {
        throw std::runtime_error("iterations must be >= 1");
    }
    if (columns.trip_distance.size() != columns.fare_amount.size()) {
        throw std::runtime_error("column size mismatch");
    }
    if (threads_per_block <= 0) {
        throw std::runtime_error("threads_per_block must be > 0");
    }

    const std::size_t n = columns.trip_distance.size();
    const std::size_t bytes_per_col = n * sizeof(float);
    const std::size_t payload_bytes = 2U * bytes_per_col;

    metrics = FilterGpuMetrics{};

    if (n == 0) {
        return Result{};
    }

    float* d_trip = nullptr;
    float* d_fare = nullptr;
    unsigned long long* d_count = nullptr;
    double* d_sum = nullptr;

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_trip), bytes_per_col), "cudaMalloc d_trip");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_fare), bytes_per_col), "cudaMalloc d_fare");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_count), sizeof(unsigned long long)), "cudaMalloc d_count");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sum), sizeof(double)), "cudaMalloc d_sum");

    cudaEvent_t e0{}, e1{}, e2{}, e3{};
    events_create(e0, e1, e2, e3);

    double acc_h2d = 0.0;
    double acc_filter = 0.0;
    double acc_d2h = 0.0;
    double acc_total = 0.0;

    const int threads = threads_per_block;
    const int blocks = static_cast<int>((n + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));

    Result last{};

    for (int it = 0; it < iterations; ++it) {
        cuda_check(cudaMemset(d_count, 0, sizeof(unsigned long long)), "cudaMemset d_count");
        cuda_check(cudaMemset(d_sum, 0, sizeof(double)), "cudaMemset d_sum");

        // total start
        cuda_check(cudaEventRecord(e0), "cudaEventRecord total_start");

        // H2D
        cuda_check(cudaEventRecord(e1), "cudaEventRecord h2d_start");
        cuda_check(cudaMemcpy(d_trip, columns.trip_distance.data(), bytes_per_col, cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D trip_distance");
        cuda_check(cudaMemcpy(d_fare, columns.fare_amount.data(), bytes_per_col, cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D fare_amount");
        cuda_check(cudaEventRecord(e2), "cudaEventRecord h2d_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize h2d_end");
        acc_h2d += elapsed_ms(e1, e2, "cudaEventElapsedTime h2d");

        // Filter kernel (atomic baseline)
        cuda_check(cudaEventRecord(e1), "cudaEventRecord filter_start");
        filter_rows_atomic_kernel<<<blocks, threads>>>(d_trip, d_fare, n, d_count, d_sum);
        cuda_check(cudaGetLastError(), "filter_rows_atomic_kernel launch");
        cuda_check(cudaEventRecord(e2), "cudaEventRecord filter_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize filter_end");
        acc_filter += elapsed_ms(e1, e2, "cudaEventElapsedTime filter");

        unsigned long long h_count = 0ULL;
        double h_sum = 0.0;
        // D2H
        cuda_check(cudaEventRecord(e1), "cudaEventRecord d2h_start");
        cuda_check(cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy D2H count");
        cuda_check(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy D2H sum");
        cuda_check(cudaEventRecord(e2), "cudaEventRecord d2h_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize d2h_end");
        acc_d2h += elapsed_ms(e1, e2, "cudaEventElapsedTime d2h");

        // total end
        cuda_check(cudaEventRecord(e3), "cudaEventRecord total_end");
        cuda_check(cudaEventSynchronize(e3), "cudaEventSynchronize total_end");
        acc_total += elapsed_ms(e0, e3, "cudaEventElapsedTime total");

        last.count = h_count;
        last.sum_fare_amount = h_sum;
    }

    events_destroy(e0, e1, e2, e3);

    cudaFree(d_trip);
    cudaFree(d_fare);
    cudaFree(d_count);
    cudaFree(d_sum);

    const double inv_it = 1.0 / static_cast<double>(iterations);
    metrics.h2d_ms = acc_h2d * inv_it;
    metrics.filter_ms = acc_filter * inv_it;
    metrics.scan_ms = 0.0;
    metrics.scatter_ms = 0.0;
    metrics.d2h_ms = acc_d2h * inv_it;
    metrics.total_gpu_ms = acc_total * inv_it;

    const double h2d_s = metrics.h2d_ms / 1000.0;
    metrics.effective_h2d_gb_per_s =
        h2d_s > 0.0 ? (static_cast<double>(payload_bytes) / h2d_s / 1e9) : 0.0;

    const double filter_s = metrics.filter_ms / 1000.0;
    metrics.rows_per_s = filter_s > 0.0 ? (static_cast<double>(n) / filter_s) : 0.0;
    metrics.filter_gb_per_s = filter_s > 0.0 ? (static_cast<double>(payload_bytes) / filter_s / 1e9) : 0.0;
    metrics.selectivity = n > 0 ? (static_cast<double>(last.count) / static_cast<double>(n)) : 0.0;

    return last;
}

FilterGpuMetrics filter_gpu_mask_only(const Columns& columns, int iterations, std::uint64_t& selected_count,
                                      int threads_per_block) {
    if (iterations < 1) {
        throw std::runtime_error("iterations must be >= 1");
    }
    if (columns.trip_distance.size() != columns.fare_amount.size()) {
        throw std::runtime_error("column size mismatch");
    }
    if (threads_per_block <= 0) {
        throw std::runtime_error("threads_per_block must be > 0");
    }

    const std::size_t n = columns.trip_distance.size();
    const std::size_t bytes_per_col = n * sizeof(float);
    const std::size_t payload_bytes = 2U * bytes_per_col;

    selected_count = 0;
    FilterGpuMetrics metrics{};
    if (n == 0) {
        return metrics;
    }

    float* d_trip = nullptr;
    float* d_fare = nullptr;
    std::uint8_t* d_mask = nullptr;

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_trip), bytes_per_col), "cudaMalloc d_trip");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_fare), bytes_per_col), "cudaMalloc d_fare");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask), n * sizeof(std::uint8_t)), "cudaMalloc d_mask");

    cudaEvent_t e0{}, e1{}, e2{}, e3{};
    events_create(e0, e1, e2, e3);

    double acc_h2d = 0.0;
    double acc_filter = 0.0;
    double acc_d2h = 0.0;
    double acc_total = 0.0;

    const int threads = threads_per_block;
    const int blocks = static_cast<int>((n + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));

    std::vector<std::uint8_t> h_mask(n);

    for (int it = 0; it < iterations; ++it) {
        cuda_check(cudaEventRecord(e0), "cudaEventRecord total_start");

        // H2D
        cuda_check(cudaEventRecord(e1), "cudaEventRecord h2d_start");
        cuda_check(cudaMemcpy(d_trip, columns.trip_distance.data(), bytes_per_col, cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D trip_distance");
        cuda_check(cudaMemcpy(d_fare, columns.fare_amount.data(), bytes_per_col, cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D fare_amount");
        cuda_check(cudaEventRecord(e2), "cudaEventRecord h2d_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize h2d_end");
        acc_h2d += elapsed_ms(e1, e2, "cudaEventElapsedTime h2d");

        // Filter
        cuda_check(cudaEventRecord(e1), "cudaEventRecord filter_start");
        filter_mask_kernel<<<blocks, threads>>>(d_trip, d_fare, n, d_mask);
        cuda_check(cudaGetLastError(), "filter_mask_kernel launch");
        cuda_check(cudaEventRecord(e2), "cudaEventRecord filter_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize filter_end");
        acc_filter += elapsed_ms(e1, e2, "cudaEventElapsedTime filter");

        // D2H (mask)
        cuda_check(cudaEventRecord(e1), "cudaEventRecord d2h_start");
        cuda_check(cudaMemcpy(h_mask.data(), d_mask, n * sizeof(std::uint8_t), cudaMemcpyDeviceToHost),
                    "cudaMemcpy D2H mask");
        cuda_check(cudaEventRecord(e2), "cudaEventRecord d2h_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize d2h_end");
        acc_d2h += elapsed_ms(e1, e2, "cudaEventElapsedTime d2h");

        cuda_check(cudaEventRecord(e3), "cudaEventRecord total_end");
        cuda_check(cudaEventSynchronize(e3), "cudaEventSynchronize total_end");
        acc_total += elapsed_ms(e0, e3, "cudaEventElapsedTime total");

        // simplest acceptable approach: count on CPU
        std::uint64_t c = 0;
        for (std::size_t i = 0; i < n; ++i) {
            c += static_cast<std::uint64_t>(h_mask[i] != 0);
        }
        selected_count = c;
    }

    events_destroy(e0, e1, e2, e3);
    cudaFree(d_trip);
    cudaFree(d_fare);
    cudaFree(d_mask);

    const double inv_it = 1.0 / static_cast<double>(iterations);
    metrics.h2d_ms = acc_h2d * inv_it;
    metrics.filter_ms = acc_filter * inv_it;
    metrics.scan_ms = 0.0;
    metrics.scatter_ms = 0.0;
    metrics.d2h_ms = acc_d2h * inv_it;
    metrics.total_gpu_ms = acc_total * inv_it;

    const double h2d_s = metrics.h2d_ms / 1000.0;
    metrics.effective_h2d_gb_per_s =
        h2d_s > 0.0 ? (static_cast<double>(payload_bytes) / h2d_s / 1e9) : 0.0;

    const double filter_s = metrics.filter_ms / 1000.0;
    metrics.rows_per_s = filter_s > 0.0 ? (static_cast<double>(n) / filter_s) : 0.0;
    metrics.filter_gb_per_s = filter_s > 0.0 ? (static_cast<double>(payload_bytes) / filter_s / 1e9) : 0.0;
    metrics.selectivity = n > 0 ? (static_cast<double>(selected_count) / static_cast<double>(n)) : 0.0;

    return metrics;
}

} // namespace gq
