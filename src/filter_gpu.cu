#include "gq/filter_gpu.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace gq {

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__global__ void filter_rows_kernel(const float* trip_distance, const float* fare_amount,
                                   std::size_t n, unsigned long long* count, double* sum_fare) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float td = trip_distance[i];
    const float fa = fare_amount[i];
    if (td > 2.5f && fa > 10.0f) {
        atomicAdd(count, 1ULL);
        atomicAdd(sum_fare, static_cast<double>(fa));
    }
}

} // namespace

Result filter_and_sum_gpu(const Columns& columns, int iterations, GpuRunMetrics& metrics) {
    if (iterations < 1) {
        throw std::runtime_error("iterations must be >= 1");
    }
    if (columns.trip_distance.size() != columns.fare_amount.size()) {
        throw std::runtime_error("column size mismatch");
    }

    const std::size_t n = columns.trip_distance.size();
    const std::size_t bytes_per_col = n * sizeof(float);
    const std::size_t h2d_payload_bytes = 2U * bytes_per_col;

    metrics = GpuRunMetrics{};

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
    cuda_check(cudaEventCreate(&e0), "cudaEventCreate e0");
    cuda_check(cudaEventCreate(&e1), "cudaEventCreate e1");
    cuda_check(cudaEventCreate(&e2), "cudaEventCreate e2");
    cuda_check(cudaEventCreate(&e3), "cudaEventCreate e3");

    double acc_h2d = 0.0;
    double acc_kernel = 0.0;
    double acc_d2h = 0.0;
    double acc_total = 0.0;

    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + static_cast<std::size_t>(threads) - 1U) / static_cast<std::size_t>(threads));

    Result last{};

    for (int it = 0; it < iterations; ++it) {
        cuda_check(cudaMemset(d_count, 0, sizeof(unsigned long long)), "cudaMemset d_count");
        cuda_check(cudaMemset(d_sum, 0, sizeof(double)), "cudaMemset d_sum");

        cuda_check(cudaEventRecord(e0), "cudaEventRecord e0");

        cuda_check(cudaMemcpy(d_trip, columns.trip_distance.data(), bytes_per_col, cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D trip_distance");
        cuda_check(cudaMemcpy(d_fare, columns.fare_amount.data(), bytes_per_col, cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D fare_amount");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord e1");
        cuda_check(cudaEventSynchronize(e1), "cudaEventSynchronize e1");
        float ms_h2d = 0.0F;
        cuda_check(cudaEventElapsedTime(&ms_h2d, e0, e1), "cudaEventElapsedTime h2d");
        acc_h2d += static_cast<double>(ms_h2d);

        filter_rows_kernel<<<blocks, threads>>>(d_trip, d_fare, n, d_count, d_sum);
        cuda_check(cudaGetLastError(), "filter_rows_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");

        cuda_check(cudaEventRecord(e2), "cudaEventRecord e2");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize e2");
        float ms_kernel = 0.0F;
        cuda_check(cudaEventElapsedTime(&ms_kernel, e1, e2), "cudaEventElapsedTime kernel");
        acc_kernel += static_cast<double>(ms_kernel);

        unsigned long long h_count = 0ULL;
        double h_sum = 0.0;
        cuda_check(cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
                    "cudaMemcpy D2H count");
        cuda_check(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy D2H sum");

        cuda_check(cudaEventRecord(e3), "cudaEventRecord e3");
        cuda_check(cudaEventSynchronize(e3), "cudaEventSynchronize e3");
        float ms_d2h = 0.0F;
        cuda_check(cudaEventElapsedTime(&ms_d2h, e2, e3), "cudaEventElapsedTime d2h");
        acc_d2h += static_cast<double>(ms_d2h);

        float ms_total = 0.0F;
        cuda_check(cudaEventElapsedTime(&ms_total, e0, e3), "cudaEventElapsedTime total");
        acc_total += static_cast<double>(ms_total);

        last.count = h_count;
        last.sum_fare_amount = h_sum;
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);

    cudaFree(d_trip);
    cudaFree(d_fare);
    cudaFree(d_count);
    cudaFree(d_sum);

    const double inv_it = 1.0 / static_cast<double>(iterations);
    metrics.h2d_ms = acc_h2d * inv_it;
    metrics.kernel_ms = acc_kernel * inv_it;
    metrics.d2h_ms = acc_d2h * inv_it;
    metrics.total_gpu_ms = acc_total * inv_it;

    const double h2d_s = metrics.h2d_ms / 1000.0;
    metrics.effective_h2d_gb_per_s =
        h2d_s > 0.0 ? (static_cast<double>(h2d_payload_bytes) / h2d_s / 1e9) : 0.0;

    return last;
}

} // namespace gq
