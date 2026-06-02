#include "gq/filter_gpu.hpp"
#include "nvtx_utils.h"

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <string>

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

inline const char* nvtx_iter_prefix_or_default(const char* prefix) {
    return prefix != nullptr ? prefix : "pipeline_iteration";
}

inline int compute_blocks(std::size_t items, int threads_per_block, int max_blocks = 4096) {
    if (items == 0) {
        return 1;
    }
    const std::size_t raw =
        (items + static_cast<std::size_t>(threads_per_block) - 1U) / static_cast<std::size_t>(threads_per_block);
    const std::size_t clamped = std::min<std::size_t>(raw, static_cast<std::size_t>(max_blocks));
    return static_cast<int>(clamped > 0 ? clamped : 1U);
}

__global__ void filter_groupby_atomic_naive_kernel(const float* trip_distance, const float* fare_amount,
                                                   const std::uint32_t* passenger_count, std::size_t n,
                                                   double* group_sum, unsigned long long* group_count,
                                                   unsigned long long* predicate_match_count,
                                                   std::uint32_t key_upper_exclusive) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = tid; i < n; i += stride) {
        const float td = trip_distance[i];
        const float fa = fare_amount[i];
        if (td > 2.5f && fa > 10.0f) {
            atomicAdd(predicate_match_count, 1ULL);
            const std::uint32_t pk = passenger_count[i];
            if (pk < key_upper_exclusive) {
                atomicAdd(&group_sum[pk], static_cast<double>(fa));
                atomicAdd(&group_count[pk], 1ULL);
            }
        }
    }
}

__global__ void groupby_selected_block_partial_kernel(const float* fare_amount, const std::uint32_t* passenger_count,
                                                      const std::uint32_t* selected_indices,
                                                      const unsigned long long* selected_count,
                                                      double* d_group_sum, unsigned long long* d_group_count,
                                                      std::uint32_t num_groups) {
    extern __shared__ unsigned char s_mem[];
    double* s_sum = reinterpret_cast<double*>(s_mem);
    unsigned long long* s_cnt =
        reinterpret_cast<unsigned long long*>(s_mem + static_cast<std::size_t>(num_groups) * sizeof(double));

    for (std::uint32_t k = static_cast<std::uint32_t>(threadIdx.x); k < num_groups;
         k += static_cast<std::uint32_t>(blockDim.x)) {
        s_sum[k] = 0.0;
        s_cnt[k] = 0ULL;
    }
    __syncthreads();

    const unsigned long long sel_count = *selected_count;
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t i = tid; i < sel_count; i += stride) {
        const std::uint32_t row = selected_indices[i];
        const std::uint32_t pk = passenger_count[row];
        if (pk < num_groups) {
            atomicAdd(&s_sum[pk], static_cast<double>(fare_amount[row]));
            atomicAdd(&s_cnt[pk], 1ULL);
        }
    }
    __syncthreads();

    for (std::uint32_t k = static_cast<std::uint32_t>(threadIdx.x); k < num_groups;
         k += static_cast<std::uint32_t>(blockDim.x)) {
        if (s_cnt[k] != 0ULL) {
            atomicAdd(&d_group_sum[k], s_sum[k]);
            atomicAdd(&d_group_count[k], s_cnt[k]);
        }
    }
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

__global__ void mask_u8_to_u64_kernel(const std::uint8_t* mask_u8, std::uint64_t* mask_u64, std::size_t n) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = tid; i < n; i += stride) {
        mask_u64[i] = static_cast<std::uint64_t>(mask_u8[i] != 0);
    }
}

__global__ void scatter_selected_indices_kernel(const std::uint64_t* mask_u64, const std::uint64_t* offsets,
                                                std::size_t n, std::uint32_t* selected_indices) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = tid; i < n; i += stride) {
        if (mask_u64[i] != 0) {
            selected_indices[offsets[i]] = static_cast<std::uint32_t>(i);
        }
    }
}

__global__ void write_selected_count_kernel(const std::uint64_t* mask_u64, const std::uint64_t* offsets, std::size_t n,
                                            unsigned long long* selected_count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *selected_count = (n == 0) ? 0ULL : static_cast<unsigned long long>(offsets[n - 1] + mask_u64[n - 1]);
    }
}

__global__ void sum_selected_fares_atomic_kernel(const float* fare_amount, const std::uint32_t* selected_indices,
                                                 const unsigned long long* selected_count, double* sum_fare) {
    const unsigned long long count = *selected_count;
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = tid; i < static_cast<std::size_t>(count); i += stride) {
        atomicAdd(sum_fare, static_cast<double>(fare_amount[selected_indices[i]]));
    }
}

__global__ void reduce_selected_fares_block_partials_kernel(const float* fare_amount, const std::uint32_t* selected_indices,
                                                            const unsigned long long* selected_count,
                                                            double* partial_sums) {
    extern __shared__ double sdata[];
    const unsigned long long count = *selected_count;
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    double local = 0.0;
    for (std::size_t i = tid; i < static_cast<std::size_t>(count); i += stride) {
        local += static_cast<double>(fare_amount[selected_indices[i]]);
    }
    sdata[threadIdx.x] = local;
    __syncthreads();

    for (unsigned int s = static_cast<unsigned int>(blockDim.x / 2); s > 0; s >>= 1U) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_partial_sums_kernel(const double* partial_sums, int num_partials, double* out_sum) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    double local = 0.0;
    for (std::size_t i = tid; i < static_cast<std::size_t>(num_partials); i += stride) {
        local += partial_sums[i];
    }
    atomicAdd(out_sum, local);
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
    metrics.reduce_ms = 0.0;
    metrics.d2h_ms = acc_d2h * inv_it;
    metrics.total_gpu_ms = acc_total * inv_it;

    const double h2d_s = metrics.h2d_ms / 1000.0;
    metrics.effective_h2d_gb_per_s =
        h2d_s > 0.0 ? (static_cast<double>(payload_bytes) / h2d_s / 1e9) : 0.0;

    const double filter_s = metrics.filter_ms / 1000.0;
    metrics.rows_per_s = filter_s > 0.0 ? (static_cast<double>(n) / filter_s) : 0.0;
    metrics.filter_gb_per_s = filter_s > 0.0 ? (static_cast<double>(payload_bytes) / filter_s / 1e9) : 0.0;
    metrics.reduce_rows_per_s = 0.0;
    metrics.reduce_gb_per_s = 0.0;
    metrics.selectivity = n > 0 ? (static_cast<double>(last.count) / static_cast<double>(n)) : 0.0;

    return last;
}

enum class ReduceAlgorithm { Atomic, BlockPartial };

Result filter_and_sum_gpu_compact_impl(const Columns& columns, int iterations, FilterGpuMetrics& metrics,
                                       int threads_per_block, ReduceAlgorithm reduce_algorithm) {
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
    std::uint8_t* d_mask = nullptr;
    std::uint64_t* d_mask_u64 = nullptr;
    std::uint64_t* d_offsets = nullptr;
    std::uint32_t* d_selected_indices = nullptr;
    unsigned long long* d_selected_count = nullptr;
    double* d_sum = nullptr;
    double* d_partial_sums = nullptr;

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_trip), bytes_per_col), "cudaMalloc d_trip");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_fare), bytes_per_col), "cudaMalloc d_fare");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask), n * sizeof(std::uint8_t)), "cudaMalloc d_mask");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask_u64), n * sizeof(std::uint64_t)), "cudaMalloc d_mask_u64");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_offsets), n * sizeof(std::uint64_t)), "cudaMalloc d_offsets");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_selected_indices), n * sizeof(std::uint32_t)),
               "cudaMalloc d_selected_indices");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_selected_count), sizeof(unsigned long long)),
               "cudaMalloc d_selected_count");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sum), sizeof(double)), "cudaMalloc d_sum");

    const int threads = threads_per_block;
    const int blocks = static_cast<int>((n + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    const int reduce_blocks = compute_blocks(n, threads);
    if (reduce_algorithm == ReduceAlgorithm::BlockPartial) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_partial_sums), static_cast<std::size_t>(reduce_blocks) * sizeof(double)),
                   "cudaMalloc d_partial_sums");
    }

    cudaEvent_t e0{}, e1{}, e2{}, e3{};
    events_create(e0, e1, e2, e3);

    double acc_h2d = 0.0;
    double acc_filter = 0.0;
    double acc_scan = 0.0;
    double acc_scatter = 0.0;
    double acc_reduce = 0.0;
    double acc_d2h = 0.0;
    double acc_total = 0.0;

    Result last{};

    for (int it = 0; it < iterations; ++it) {
        cuda_check(cudaMemset(d_selected_count, 0, sizeof(unsigned long long)), "cudaMemset d_selected_count");
        cuda_check(cudaMemset(d_sum, 0, sizeof(double)), "cudaMemset d_sum");
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

        // Prefix-sum over mask to build write offsets.
        cuda_check(cudaEventRecord(e1), "cudaEventRecord scan_start");
        mask_u8_to_u64_kernel<<<blocks, threads>>>(d_mask, d_mask_u64, n);
        cuda_check(cudaGetLastError(), "mask_u8_to_u64_kernel launch");
        thrust::exclusive_scan(thrust::device, d_mask_u64, d_mask_u64 + n, d_offsets);
        cuda_check(cudaGetLastError(), "thrust::exclusive_scan");
        cuda_check(cudaEventRecord(e2), "cudaEventRecord scan_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize scan_end");
        acc_scan += elapsed_ms(e1, e2, "cudaEventElapsedTime scan");

        // Scatter selected row indices into compact output.
        cuda_check(cudaEventRecord(e1), "cudaEventRecord scatter_start");
        scatter_selected_indices_kernel<<<blocks, threads>>>(d_mask_u64, d_offsets, n, d_selected_indices);
        cuda_check(cudaGetLastError(), "scatter_selected_indices_kernel launch");
        write_selected_count_kernel<<<1, 1>>>(d_mask_u64, d_offsets, n, d_selected_count);
        cuda_check(cudaGetLastError(), "write_selected_count_kernel launch");
        cuda_check(cudaEventRecord(e2), "cudaEventRecord scatter_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize scatter_end");
        acc_scatter += elapsed_ms(e1, e2, "cudaEventElapsedTime scatter");

        // Reduce selected fares on GPU
        cuda_check(cudaEventRecord(e1), "cudaEventRecord reduce_start");
        if (reduce_algorithm == ReduceAlgorithm::Atomic) {
            sum_selected_fares_atomic_kernel<<<reduce_blocks, threads>>>(d_fare, d_selected_indices, d_selected_count, d_sum);
            cuda_check(cudaGetLastError(), "sum_selected_fares_atomic_kernel launch");
        } else {
            const std::size_t shmem_bytes = static_cast<std::size_t>(threads) * sizeof(double);
            reduce_selected_fares_block_partials_kernel<<<reduce_blocks, threads, shmem_bytes>>>(
                d_fare, d_selected_indices, d_selected_count, d_partial_sums);
            cuda_check(cudaGetLastError(), "reduce_selected_fares_block_partials_kernel launch");
            reduce_partial_sums_kernel<<<reduce_blocks, threads>>>(d_partial_sums, reduce_blocks, d_sum);
            cuda_check(cudaGetLastError(), "reduce_partial_sums_kernel launch");
        }
        cuda_check(cudaEventRecord(e2), "cudaEventRecord reduce_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize reduce_end");
        acc_reduce += elapsed_ms(e1, e2, "cudaEventElapsedTime reduce");

        // D2H final results only
        cuda_check(cudaEventRecord(e1), "cudaEventRecord d2h_start");
        cuda_check(cudaMemcpy(&last.count, d_selected_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H selected_count");
        cuda_check(cudaMemcpy(&last.sum_fare_amount, d_sum, sizeof(double), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H sum");
        cuda_check(cudaEventRecord(e2), "cudaEventRecord d2h_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize d2h_end");
        acc_d2h += elapsed_ms(e1, e2, "cudaEventElapsedTime d2h");

        cuda_check(cudaEventRecord(e3), "cudaEventRecord total_end");
        cuda_check(cudaEventSynchronize(e3), "cudaEventSynchronize total_end");
        acc_total += elapsed_ms(e0, e3, "cudaEventElapsedTime total");
    }

    events_destroy(e0, e1, e2, e3);
    cudaFree(d_trip);
    cudaFree(d_fare);
    cudaFree(d_mask);
    cudaFree(d_mask_u64);
    cudaFree(d_offsets);
    cudaFree(d_selected_indices);
    cudaFree(d_selected_count);
    cudaFree(d_sum);
    if (d_partial_sums != nullptr) {
        cudaFree(d_partial_sums);
    }

    const double inv_it = 1.0 / static_cast<double>(iterations);
    metrics.h2d_ms = acc_h2d * inv_it;
    metrics.filter_ms = acc_filter * inv_it;
    metrics.scan_ms = acc_scan * inv_it;
    metrics.scatter_ms = acc_scatter * inv_it;
    metrics.reduce_ms = acc_reduce * inv_it;
    metrics.d2h_ms = acc_d2h * inv_it;
    metrics.total_gpu_ms = acc_total * inv_it;

    const double h2d_s = metrics.h2d_ms / 1000.0;
    metrics.effective_h2d_gb_per_s =
        h2d_s > 0.0 ? (static_cast<double>(payload_bytes) / h2d_s / 1e9) : 0.0;

    const double filter_s = metrics.filter_ms / 1000.0;
    metrics.rows_per_s = filter_s > 0.0 ? (static_cast<double>(n) / filter_s) : 0.0;
    metrics.filter_gb_per_s = filter_s > 0.0 ? (static_cast<double>(payload_bytes) / filter_s / 1e9) : 0.0;
    const double reduce_s = metrics.reduce_ms / 1000.0;
    metrics.reduce_rows_per_s = reduce_s > 0.0 ? (static_cast<double>(last.count) / reduce_s) : 0.0;
    metrics.reduce_gb_per_s = reduce_s > 0.0
                                  ? ((static_cast<double>(last.count) * sizeof(float)) / reduce_s / 1e9)
                                  : 0.0;
    metrics.selectivity = n > 0 ? (static_cast<double>(last.count) / static_cast<double>(n)) : 0.0;

    return last;
}

Result filter_and_sum_gpu_compact_atomic(const Columns& columns, int iterations, FilterGpuMetrics& metrics,
                                         int threads_per_block) {
    return filter_and_sum_gpu_compact_impl(columns, iterations, metrics, threads_per_block, ReduceAlgorithm::Atomic);
}

Result filter_and_sum_gpu_compact_block_partial(const Columns& columns, int iterations, FilterGpuMetrics& metrics,
                                                int threads_per_block) {
    return filter_and_sum_gpu_compact_impl(columns, iterations, metrics, threads_per_block,
                                           ReduceAlgorithm::BlockPartial);
}

GroupByGpuTable filter_groupby_gpu_atomic_baseline(const Columns& columns, int iterations, FilterGpuMetrics& metrics,
                                                   std::size_t num_groups, int threads_per_block,
                                                   const char* nvtx_iteration_prefix) {
    if (iterations < 1) {
        throw std::runtime_error("iterations must be >= 1");
    }
    if (columns.trip_distance.size() != columns.fare_amount.size()) {
        throw std::runtime_error("column size mismatch");
    }
    if (columns.passenger_count.size() != columns.trip_distance.size()) {
        throw std::runtime_error("passenger_count column size mismatch");
    }
    if (num_groups == 0) {
        throw std::runtime_error("num_groups must be > 0");
    }
    if (threads_per_block <= 0) {
        throw std::runtime_error("threads_per_block must be > 0");
    }

    const std::size_t n = columns.trip_distance.size();
    const std::size_t bytes_float_col = n * sizeof(float);
    const std::size_t bytes_pc = n * sizeof(std::uint32_t);
    const std::size_t payload_bytes = 2U * bytes_float_col + bytes_pc;

    metrics = FilterGpuMetrics{};
    GroupByGpuTable last{};
    last.sum.assign(num_groups, 0.0);
    last.count.assign(num_groups, 0U);

    if (n == 0) {
        return last;
    }

    float* d_trip = nullptr;
    float* d_fare = nullptr;
    std::uint32_t* d_pc = nullptr;
    double* d_group_sum = nullptr;
    unsigned long long* d_group_count = nullptr;
    unsigned long long* d_pred_match = nullptr;

    {
        NvtxRange alloc_range("cuda_allocation");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_trip), bytes_float_col), "cudaMalloc d_trip");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_fare), bytes_float_col), "cudaMalloc d_fare");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pc), bytes_pc), "cudaMalloc d_pc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group_sum),
                              static_cast<std::size_t>(num_groups) * sizeof(double)),
                   "cudaMalloc d_group_sum");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group_count),
                              static_cast<std::size_t>(num_groups) * sizeof(unsigned long long)),
                   "cudaMalloc d_group_count");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pred_match), sizeof(unsigned long long)),
                   "cudaMalloc d_pred_match");
    }

    cudaEvent_t e0{}, e1{}, e2{}, e3{};
    events_create(e0, e1, e2, e3);

    double acc_h2d = 0.0;
    double acc_groupby = 0.0;
    double acc_d2h = 0.0;
    double acc_total = 0.0;

    const int threads = threads_per_block;
    const int blocks = static_cast<int>((n + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));

    std::uint64_t selected_for_selectivity = 0;
    const char* iter_prefix = nvtx_iter_prefix_or_default(nvtx_iteration_prefix);

    for (int it = 0; it < iterations; ++it) {
        NvtxRange iteration_range(iter_prefix, it);

        cuda_check(cudaMemset(d_group_sum, 0, static_cast<std::size_t>(num_groups) * sizeof(double)),
                    "cudaMemset d_group_sum");
        cuda_check(cudaMemset(d_group_count, 0,
                              static_cast<std::size_t>(num_groups) * sizeof(unsigned long long)),
                    "cudaMemset d_group_count");
        cuda_check(cudaMemset(d_pred_match, 0, sizeof(unsigned long long)), "cudaMemset d_pred_match");

        cuda_check(cudaEventRecord(e0), "cudaEventRecord total_start");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord h2d_start");
        {
            NvtxRange h2d_range("h2d_transfer");
            cuda_check(cudaMemcpy(d_trip, columns.trip_distance.data(), bytes_float_col, cudaMemcpyHostToDevice),
                        "cudaMemcpy H2D trip_distance");
            cuda_check(cudaMemcpy(d_fare, columns.fare_amount.data(), bytes_float_col, cudaMemcpyHostToDevice),
                        "cudaMemcpy H2D fare_amount");
            cuda_check(cudaMemcpy(d_pc, columns.passenger_count.data(), bytes_pc, cudaMemcpyHostToDevice),
                        "cudaMemcpy H2D passenger_count");
        }
        cuda_check(cudaEventRecord(e2), "cudaEventRecord h2d_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize h2d_end");
        acc_h2d += elapsed_ms(e1, e2, "cudaEventElapsedTime h2d");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord groupby_start");
        {
            NvtxRange kernel_range("atomic_groupby_sum_kernel_launch");
            filter_groupby_atomic_naive_kernel<<<blocks, threads>>>(
                d_trip, d_fare, d_pc, n, d_group_sum, d_group_count, d_pred_match,
                static_cast<std::uint32_t>(num_groups));
            cuda_check(cudaGetLastError(), "filter_groupby_atomic_naive_kernel launch");
        }
        cuda_check(cudaEventRecord(e2), "cudaEventRecord groupby_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize groupby_end");
        acc_groupby += elapsed_ms(e1, e2, "cudaEventElapsedTime groupby");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord d2h_start");
        {
            NvtxRange d2h_range("d2h_transfer");
            cuda_check(cudaMemcpy(last.sum.data(), d_group_sum, static_cast<std::size_t>(num_groups) * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H group_sum");
            cuda_check(cudaMemcpy(last.count.data(), d_group_count,
                                  static_cast<std::size_t>(num_groups) * sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H group_count");
            cuda_check(cudaMemcpy(&selected_for_selectivity, d_pred_match, sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H predicate_match_count");
        }
        cuda_check(cudaEventRecord(e2), "cudaEventRecord d2h_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize d2h_end");
        acc_d2h += elapsed_ms(e1, e2, "cudaEventElapsedTime d2h");

        cuda_check(cudaEventRecord(e3), "cudaEventRecord total_end");
        cuda_check(cudaEventSynchronize(e3), "cudaEventSynchronize total_end");
        acc_total += elapsed_ms(e0, e3, "cudaEventElapsedTime total");
    }

    events_destroy(e0, e1, e2, e3);
    {
        NvtxRange free_range("cuda_free");
        cudaFree(d_trip);
        cudaFree(d_fare);
        cudaFree(d_pc);
        cudaFree(d_group_sum);
        cudaFree(d_group_count);
        cudaFree(d_pred_match);
    }

    const double inv_it = 1.0 / static_cast<double>(iterations);
    metrics.h2d_ms = acc_h2d * inv_it;
    metrics.filter_ms = 0.0;
    metrics.scan_ms = 0.0;
    metrics.scatter_ms = 0.0;
    metrics.reduce_ms = 0.0;
    metrics.groupby_ms = acc_groupby * inv_it;
    metrics.d2h_ms = acc_d2h * inv_it;
    metrics.total_gpu_ms = acc_total * inv_it;

    const double h2d_s = metrics.h2d_ms / 1000.0;
    metrics.effective_h2d_gb_per_s =
        h2d_s > 0.0 ? (static_cast<double>(payload_bytes) / h2d_s / 1e9) : 0.0;

    const double groupby_s = metrics.groupby_ms / 1000.0;
    metrics.rows_per_s = groupby_s > 0.0 ? (static_cast<double>(n) / groupby_s) : 0.0;
    metrics.filter_gb_per_s = 0.0;
    metrics.reduce_rows_per_s = 0.0;
    metrics.reduce_gb_per_s = 0.0;
    metrics.groupby_rows_per_s = metrics.rows_per_s;
    metrics.groupby_gb_per_s =
        groupby_s > 0.0 ? (static_cast<double>(payload_bytes) / groupby_s / 1e9) : 0.0;
    metrics.selectivity =
        n > 0 ? (static_cast<double>(selected_for_selectivity) / static_cast<double>(n)) : 0.0;
    metrics.predicate_selected_count = selected_for_selectivity;

    return last;
}

GroupByGpuTable filter_groupby_gpu_compact_block_partial(const Columns& columns, int iterations, FilterGpuMetrics& metrics,
                                                         std::size_t num_groups, int threads_per_block,
                                                         const char* nvtx_iteration_prefix) {
    if (iterations < 1) {
        throw std::runtime_error("iterations must be >= 1");
    }
    if (columns.trip_distance.size() != columns.fare_amount.size()) {
        throw std::runtime_error("column size mismatch");
    }
    if (columns.passenger_count.size() != columns.trip_distance.size()) {
        throw std::runtime_error("passenger_count column size mismatch");
    }
    if (num_groups == 0) {
        throw std::runtime_error("num_groups must be > 0");
    }
    if (threads_per_block <= 0) {
        throw std::runtime_error("threads_per_block must be > 0");
    }

    const std::size_t n = columns.trip_distance.size();
    const std::size_t bytes_float_col = n * sizeof(float);
    const std::size_t bytes_pc = n * sizeof(std::uint32_t);
    const std::size_t payload_bytes = 2U * bytes_float_col + bytes_pc;

    metrics = FilterGpuMetrics{};
    GroupByGpuTable last{};
    last.sum.assign(num_groups, 0.0);
    last.count.assign(num_groups, 0U);

    if (n == 0) {
        return last;
    }

    float* d_trip = nullptr;
    float* d_fare = nullptr;
    std::uint32_t* d_pc = nullptr;
    std::uint8_t* d_mask = nullptr;
    std::uint64_t* d_mask_u64 = nullptr;
    std::uint64_t* d_offsets = nullptr;
    std::uint32_t* d_selected_indices = nullptr;
    unsigned long long* d_selected_count = nullptr;
    double* d_group_sum = nullptr;
    unsigned long long* d_group_count = nullptr;

    {
        NvtxRange alloc_range("cuda_allocation");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_trip), bytes_float_col), "cudaMalloc d_trip");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_fare), bytes_float_col), "cudaMalloc d_fare");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pc), bytes_pc), "cudaMalloc d_pc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask), n * sizeof(std::uint8_t)), "cudaMalloc d_mask");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask_u64), n * sizeof(std::uint64_t)),
                   "cudaMalloc d_mask_u64");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_offsets), n * sizeof(std::uint64_t)), "cudaMalloc d_offsets");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_selected_indices), n * sizeof(std::uint32_t)),
                   "cudaMalloc d_selected_indices");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_selected_count), sizeof(unsigned long long)),
                   "cudaMalloc d_selected_count");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group_sum),
                              static_cast<std::size_t>(num_groups) * sizeof(double)),
                   "cudaMalloc d_group_sum");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group_count),
                              static_cast<std::size_t>(num_groups) * sizeof(unsigned long long)),
                   "cudaMalloc d_group_count");
    }

    const int threads = threads_per_block;
    const int blocks = static_cast<int>((n + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    const int reduce_blocks = compute_blocks(n, threads);

    cudaEvent_t e0{}, e1{}, e2{}, e3{};
    events_create(e0, e1, e2, e3);

    double acc_h2d = 0.0;
    double acc_filter = 0.0;
    double acc_scan = 0.0;
    double acc_scatter = 0.0;
    double acc_groupby = 0.0;
    double acc_d2h = 0.0;
    double acc_total = 0.0;

    unsigned long long last_selected = 0;
    const char* iter_prefix = nvtx_iter_prefix_or_default(nvtx_iteration_prefix);

    for (int it = 0; it < iterations; ++it) {
        NvtxRange iteration_range(iter_prefix, it);

        cuda_check(cudaMemset(d_selected_count, 0, sizeof(unsigned long long)), "cudaMemset d_selected_count");
        cuda_check(cudaMemset(d_group_sum, 0, static_cast<std::size_t>(num_groups) * sizeof(double)),
                    "cudaMemset d_group_sum");
        cuda_check(cudaMemset(d_group_count, 0,
                              static_cast<std::size_t>(num_groups) * sizeof(unsigned long long)),
                    "cudaMemset d_group_count");

        cuda_check(cudaEventRecord(e0), "cudaEventRecord total_start");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord h2d_start");
        {
            NvtxRange h2d_range("h2d_transfer");
            cuda_check(cudaMemcpy(d_trip, columns.trip_distance.data(), bytes_float_col, cudaMemcpyHostToDevice),
                        "cudaMemcpy H2D trip_distance");
            cuda_check(cudaMemcpy(d_fare, columns.fare_amount.data(), bytes_float_col, cudaMemcpyHostToDevice),
                        "cudaMemcpy H2D fare_amount");
            cuda_check(cudaMemcpy(d_pc, columns.passenger_count.data(), bytes_pc, cudaMemcpyHostToDevice),
                        "cudaMemcpy H2D passenger_count");
        }
        cuda_check(cudaEventRecord(e2), "cudaEventRecord h2d_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize h2d_end");
        acc_h2d += elapsed_ms(e1, e2, "cudaEventElapsedTime h2d");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord filter_start");
        {
            NvtxRange filter_range("filter_kernel_launch");
            filter_mask_kernel<<<blocks, threads>>>(d_trip, d_fare, n, d_mask);
            cuda_check(cudaGetLastError(), "filter_mask_kernel launch");
        }
        cuda_check(cudaEventRecord(e2), "cudaEventRecord filter_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize filter_end");
        acc_filter += elapsed_ms(e1, e2, "cudaEventElapsedTime filter");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord scan_start");
        mask_u8_to_u64_kernel<<<blocks, threads>>>(d_mask, d_mask_u64, n);
        cuda_check(cudaGetLastError(), "mask_u8_to_u64_kernel launch");
        thrust::exclusive_scan(thrust::device, d_mask_u64, d_mask_u64 + n, d_offsets);
        cuda_check(cudaGetLastError(), "thrust::exclusive_scan");
        cuda_check(cudaEventRecord(e2), "cudaEventRecord scan_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize scan_end");
        acc_scan += elapsed_ms(e1, e2, "cudaEventElapsedTime scan");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord scatter_start");
        scatter_selected_indices_kernel<<<blocks, threads>>>(d_mask_u64, d_offsets, n, d_selected_indices);
        cuda_check(cudaGetLastError(), "scatter_selected_indices_kernel launch");
        write_selected_count_kernel<<<1, 1>>>(d_mask_u64, d_offsets, n, d_selected_count);
        cuda_check(cudaGetLastError(), "write_selected_count_kernel launch");
        cuda_check(cudaEventRecord(e2), "cudaEventRecord scatter_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize scatter_end");
        acc_scatter += elapsed_ms(e1, e2, "cudaEventElapsedTime scatter");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord groupby_start");
        {
            NvtxRange reduce_range("block_partial_reduce_kernel_launch");
            const std::size_t shmem_bytes =
                static_cast<std::size_t>(num_groups) * (sizeof(double) + sizeof(unsigned long long));
            groupby_selected_block_partial_kernel<<<reduce_blocks, threads, shmem_bytes>>>(
                d_fare, d_pc, d_selected_indices, d_selected_count, d_group_sum, d_group_count,
                static_cast<std::uint32_t>(num_groups));
            cuda_check(cudaGetLastError(), "groupby_selected_block_partial_kernel launch");
        }
        cuda_check(cudaEventRecord(e2), "cudaEventRecord groupby_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize groupby_end");
        acc_groupby += elapsed_ms(e1, e2, "cudaEventElapsedTime groupby");

        cuda_check(cudaEventRecord(e1), "cudaEventRecord d2h_start");
        {
            NvtxRange d2h_range("d2h_transfer");
            cuda_check(cudaMemcpy(&last_selected, d_selected_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
                        "cudaMemcpy D2H selected_count");
            cuda_check(cudaMemcpy(last.sum.data(), d_group_sum, static_cast<std::size_t>(num_groups) * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H group_sum");
            cuda_check(cudaMemcpy(last.count.data(), d_group_count,
                                  static_cast<std::size_t>(num_groups) * sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H group_count");
        }
        cuda_check(cudaEventRecord(e2), "cudaEventRecord d2h_end");
        cuda_check(cudaEventSynchronize(e2), "cudaEventSynchronize d2h_end");
        acc_d2h += elapsed_ms(e1, e2, "cudaEventElapsedTime d2h");

        cuda_check(cudaEventRecord(e3), "cudaEventRecord total_end");
        cuda_check(cudaEventSynchronize(e3), "cudaEventSynchronize total_end");
        acc_total += elapsed_ms(e0, e3, "cudaEventElapsedTime total");
    }

    events_destroy(e0, e1, e2, e3);
    {
        NvtxRange free_range("cuda_free");
        cudaFree(d_trip);
        cudaFree(d_fare);
        cudaFree(d_pc);
        cudaFree(d_mask);
        cudaFree(d_mask_u64);
        cudaFree(d_offsets);
        cudaFree(d_selected_indices);
        cudaFree(d_selected_count);
        cudaFree(d_group_sum);
        cudaFree(d_group_count);
    }

    const double inv_it = 1.0 / static_cast<double>(iterations);
    metrics.h2d_ms = acc_h2d * inv_it;
    metrics.filter_ms = acc_filter * inv_it;
    metrics.scan_ms = acc_scan * inv_it;
    metrics.scatter_ms = acc_scatter * inv_it;
    metrics.reduce_ms = 0.0;
    metrics.groupby_ms = acc_groupby * inv_it;
    metrics.d2h_ms = acc_d2h * inv_it;
    metrics.total_gpu_ms = acc_total * inv_it;

    const double h2d_s = metrics.h2d_ms / 1000.0;
    metrics.effective_h2d_gb_per_s =
        h2d_s > 0.0 ? (static_cast<double>(payload_bytes) / h2d_s / 1e9) : 0.0;

    const double filter_s = metrics.filter_ms / 1000.0;
    metrics.rows_per_s = filter_s > 0.0 ? (static_cast<double>(n) / filter_s) : 0.0;
    metrics.filter_gb_per_s = filter_s > 0.0 ? (static_cast<double>(payload_bytes) / filter_s / 1e9) : 0.0;

    const double groupby_s = metrics.groupby_ms / 1000.0;
    metrics.groupby_rows_per_s =
        groupby_s > 0.0 ? (static_cast<double>(last_selected) / groupby_s) : 0.0;
    metrics.groupby_gb_per_s = groupby_s > 0.0
                                   ? (static_cast<double>(last_selected) * (sizeof(float) + sizeof(std::uint8_t)) /
                                      groupby_s / 1e9)
                                   : 0.0;
    metrics.reduce_rows_per_s = 0.0;
    metrics.reduce_gb_per_s = 0.0;
    metrics.selectivity = n > 0 ? (static_cast<double>(last_selected) / static_cast<double>(n)) : 0.0;
    metrics.predicate_selected_count = last_selected;

    return last;
}

} // namespace gq
