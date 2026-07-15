#include "gq/filter_gpu.hpp"
#include "gq/pipeline_timing.hpp"
#include "gq/timer.hpp"
#include "nvtx_utils.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>
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

inline void event_create(cudaEvent_t& e) {
    cuda_check(cudaEventCreate(&e), "cudaEventCreate");
}

inline void event_destroy(cudaEvent_t e) {
    cudaEventDestroy(e);
}

inline void synchronize_pipeline(cudaStream_t stream, cudaEvent_t total_stop) {
    if (stream != nullptr) {
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    cuda_check(cudaEventSynchronize(total_stop), "cudaEventSynchronize total_stop");
}

inline void record_pipeline_timing_atomic(cudaEvent_t total_start, cudaEvent_t total_stop, cudaEvent_t h2d_start,
                                          cudaEvent_t h2d_stop, cudaEvent_t atomic_start, cudaEvent_t atomic_stop,
                                          cudaEvent_t d2h_start, cudaEvent_t d2h_stop, PipelineTiming& timing,
                                          cudaStream_t stream) {
    synchronize_pipeline(stream, total_stop);
    timing.total_gpu_ms = elapsed_ms(total_start, total_stop, "cudaEventElapsedTime total");
    timing.h2d_ms = elapsed_ms(h2d_start, h2d_stop, "cudaEventElapsedTime h2d");
    timing.filter_kernel_ms = 0.0;
    timing.scan_kernel_ms = 0.0;
    timing.scatter_kernel_ms = 0.0;
    timing.atomic_groupby_kernel_ms = elapsed_ms(atomic_start, atomic_stop, "cudaEventElapsedTime atomic_groupby");
    timing.block_partial_reduce_kernel_ms = 0.0;
    timing.final_reduce_kernel_ms = 0.0;
    timing.d2h_ms = elapsed_ms(d2h_start, d2h_stop, "cudaEventElapsedTime d2h");
}

inline void record_pipeline_timing_block_partial(cudaEvent_t total_start, cudaEvent_t total_stop, cudaEvent_t h2d_start,
                                                 cudaEvent_t h2d_stop, cudaEvent_t filter_start, cudaEvent_t filter_stop,
                                                 cudaEvent_t scan_start, cudaEvent_t scan_stop, cudaEvent_t scatter_start,
                                                 cudaEvent_t scatter_stop, cudaEvent_t reduce_start, cudaEvent_t reduce_stop,
                                                 cudaEvent_t d2h_start, cudaEvent_t d2h_stop, PipelineTiming& timing,
                                                 cudaStream_t stream) {
    synchronize_pipeline(stream, total_stop);
    timing.total_gpu_ms = elapsed_ms(total_start, total_stop, "cudaEventElapsedTime total");
    timing.h2d_ms = elapsed_ms(h2d_start, h2d_stop, "cudaEventElapsedTime h2d");
    timing.filter_kernel_ms = elapsed_ms(filter_start, filter_stop, "cudaEventElapsedTime filter");
    timing.scan_kernel_ms = elapsed_ms(scan_start, scan_stop, "cudaEventElapsedTime scan");
    timing.scatter_kernel_ms = elapsed_ms(scatter_start, scatter_stop, "cudaEventElapsedTime scatter");
    timing.atomic_groupby_kernel_ms = 0.0;
    timing.block_partial_reduce_kernel_ms =
        elapsed_ms(reduce_start, reduce_stop, "cudaEventElapsedTime block_partial_reduce");
    timing.final_reduce_kernel_ms = 0.0;
    timing.d2h_ms = elapsed_ms(d2h_start, d2h_stop, "cudaEventElapsedTime d2h");
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

class CudaStreamGuard {
public:
    explicit CudaStreamGuard(ExecutionMode execution) : async_(execution == ExecutionMode::SingleStreamAsync) {
        if (async_) {
            cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
        }
    }

    ~CudaStreamGuard() {
        if (async_) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStreamGuard(const CudaStreamGuard&) = delete;
    CudaStreamGuard& operator=(const CudaStreamGuard&) = delete;

    [[nodiscard]] bool async() const noexcept {
        return async_;
    }

    [[nodiscard]] cudaStream_t stream() const noexcept {
        return async_ ? stream_ : nullptr;
    }

private:
    bool async_{false};
    cudaStream_t stream_{};
};

inline void event_record(cudaEvent_t event, cudaStream_t stream) {
    if (stream != nullptr) {
        cuda_check(cudaEventRecord(event, stream), "cudaEventRecord");
    } else {
        cuda_check(cudaEventRecord(event), "cudaEventRecord");
    }
}

inline void memcpy_h2d(void* dst, const void* src, std::size_t bytes, cudaStream_t stream) {
    if (stream != nullptr) {
        cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");
    } else {
        cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
}

inline void memcpy_d2h(void* dst, const void* src, std::size_t bytes, cudaStream_t stream) {
    if (stream != nullptr) {
        cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
    } else {
        cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }
}

inline void memset_device(void* dst, int value, std::size_t bytes, cudaStream_t stream) {
    if (stream != nullptr) {
        cuda_check(cudaMemsetAsync(dst, value, bytes, stream), "cudaMemsetAsync");
    } else {
        cuda_check(cudaMemset(dst, value, bytes), "cudaMemset");
    }
}

inline void compute_batch_plan(std::size_t num_rows, std::size_t batch_rows, std::size_t& effective_batch_rows,
                               std::size_t& num_batches) {
    if (batch_rows == 0 || batch_rows >= num_rows) {
        effective_batch_rows = num_rows;
        num_batches = 1;
    } else {
        effective_batch_rows = batch_rows;
        num_batches = (num_rows + batch_rows - 1) / batch_rows;
    }
}

inline void merge_partial_groupby_cpu(const std::vector<double>& partial_sums,
                                      const std::vector<unsigned long long>& partial_counts, std::size_t num_batches,
                                      std::size_t num_groups, GroupByGpuTable& out) {
    out.sum.assign(num_groups, 0.0);
    out.count.assign(num_groups, 0U);
    for (std::size_t b = 0; b < num_batches; ++b) {
        for (std::size_t g = 0; g < num_groups; ++g) {
            const std::size_t idx = b * num_groups + g;
            out.sum[g] += partial_sums[idx];
            out.count[g] += static_cast<std::uint64_t>(partial_counts[idx]);
        }
    }
}

inline void add_pipeline_timing(PipelineTiming& dst, const PipelineTiming& src) {
    dst.total_gpu_ms += src.total_gpu_ms;
    dst.h2d_ms += src.h2d_ms;
    dst.filter_kernel_ms += src.filter_kernel_ms;
    dst.scan_kernel_ms += src.scan_kernel_ms;
    dst.scatter_kernel_ms += src.scatter_kernel_ms;
    dst.atomic_groupby_kernel_ms += src.atomic_groupby_kernel_ms;
    dst.block_partial_reduce_kernel_ms += src.block_partial_reduce_kernel_ms;
    dst.final_reduce_kernel_ms += src.final_reduce_kernel_ms;
    dst.d2h_ms += src.d2h_ms;
}

struct AtomicStreamContext {
    cudaStream_t stream = nullptr;
    float* d_trip = nullptr;
    float* d_fare = nullptr;
    std::uint32_t* d_pc = nullptr;
    double* d_group_sum = nullptr;
    unsigned long long* d_group_count = nullptr;
    unsigned long long* d_pred_match = nullptr;

    void allocate(std::size_t max_batch_rows, std::size_t num_groups) {
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_trip), max_batch_rows * sizeof(float)), "cudaMalloc d_trip");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_fare), max_batch_rows * sizeof(float)), "cudaMalloc d_fare");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pc), max_batch_rows * sizeof(std::uint32_t)),
                   "cudaMalloc d_pc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group_sum), num_groups * sizeof(double)),
                   "cudaMalloc d_group_sum");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group_count), num_groups * sizeof(unsigned long long)),
                   "cudaMalloc d_group_count");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pred_match), sizeof(unsigned long long)),
                   "cudaMalloc d_pred_match");
    }

    void free() {
        if (d_trip != nullptr) {
            cudaFree(d_trip);
            d_trip = nullptr;
        }
        if (d_fare != nullptr) {
            cudaFree(d_fare);
            d_fare = nullptr;
        }
        if (d_pc != nullptr) {
            cudaFree(d_pc);
            d_pc = nullptr;
        }
        if (d_group_sum != nullptr) {
            cudaFree(d_group_sum);
            d_group_sum = nullptr;
        }
        if (d_group_count != nullptr) {
            cudaFree(d_group_count);
            d_group_count = nullptr;
        }
        if (d_pred_match != nullptr) {
            cudaFree(d_pred_match);
            d_pred_match = nullptr;
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }
};

struct BlockPartialStreamContext {
    cudaStream_t stream = nullptr;
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

    void allocate(std::size_t max_batch_rows, std::size_t num_groups) {
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_trip), max_batch_rows * sizeof(float)), "cudaMalloc d_trip");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_fare), max_batch_rows * sizeof(float)), "cudaMalloc d_fare");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pc), max_batch_rows * sizeof(std::uint32_t)),
                   "cudaMalloc d_pc");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask), max_batch_rows * sizeof(std::uint8_t)),
                   "cudaMalloc d_mask");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask_u64), max_batch_rows * sizeof(std::uint64_t)),
                   "cudaMalloc d_mask_u64");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_offsets), max_batch_rows * sizeof(std::uint64_t)),
                   "cudaMalloc d_offsets");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_selected_indices), max_batch_rows * sizeof(std::uint32_t)),
                   "cudaMalloc d_selected_indices");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_selected_count), sizeof(unsigned long long)),
                   "cudaMalloc d_selected_count");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group_sum), num_groups * sizeof(double)),
                   "cudaMalloc d_group_sum");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group_count), num_groups * sizeof(unsigned long long)),
                   "cudaMalloc d_group_count");
    }

    void free() {
        if (d_trip != nullptr) {
            cudaFree(d_trip);
            d_trip = nullptr;
        }
        if (d_fare != nullptr) {
            cudaFree(d_fare);
            d_fare = nullptr;
        }
        if (d_pc != nullptr) {
            cudaFree(d_pc);
            d_pc = nullptr;
        }
        if (d_mask != nullptr) {
            cudaFree(d_mask);
            d_mask = nullptr;
        }
        if (d_mask_u64 != nullptr) {
            cudaFree(d_mask_u64);
            d_mask_u64 = nullptr;
        }
        if (d_offsets != nullptr) {
            cudaFree(d_offsets);
            d_offsets = nullptr;
        }
        if (d_selected_indices != nullptr) {
            cudaFree(d_selected_indices);
            d_selected_indices = nullptr;
        }
        if (d_selected_count != nullptr) {
            cudaFree(d_selected_count);
            d_selected_count = nullptr;
        }
        if (d_group_sum != nullptr) {
            cudaFree(d_group_sum);
            d_group_sum = nullptr;
        }
        if (d_group_count != nullptr) {
            cudaFree(d_group_count);
            d_group_count = nullptr;
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }
};

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
                                                   const char* nvtx_iteration_prefix, TimingStats* measured_timing,
                                                   ExecutionMode execution, std::size_t batch_rows, int num_streams) {
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
    if (num_streams < 1) {
        throw std::runtime_error("num_streams must be >= 1");
    }

    const std::size_t n = columns.trip_distance.size();
    std::size_t effective_batch_rows = 0;
    std::size_t num_batches = 0;
    compute_batch_plan(n, batch_rows, effective_batch_rows, num_batches);
    const std::size_t max_batch_rows = effective_batch_rows;
    const std::size_t payload_bytes = n * (2U * sizeof(float) + sizeof(std::uint32_t));
    const bool multi_stream = (execution == ExecutionMode::MultiStreamBatched);
    const int active_streams = multi_stream ? num_streams : 1;

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
    std::vector<AtomicStreamContext> stream_contexts;

    {
        NvtxRange alloc_range("cuda_allocation");
        if (multi_stream) {
            stream_contexts.resize(static_cast<std::size_t>(active_streams));
            for (auto& ctx : stream_contexts) {
                ctx.allocate(max_batch_rows, num_groups);
            }
        } else {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_trip), max_batch_rows * sizeof(float)),
                       "cudaMalloc d_trip");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_fare), max_batch_rows * sizeof(float)),
                       "cudaMalloc d_fare");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pc), max_batch_rows * sizeof(std::uint32_t)),
                       "cudaMalloc d_pc");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group_sum),
                                  static_cast<std::size_t>(num_groups) * sizeof(double)),
                       "cudaMalloc d_group_sum");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group_count),
                                  static_cast<std::size_t>(num_groups) * sizeof(unsigned long long)),
                       "cudaMalloc d_group_count");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pred_match), sizeof(unsigned long long)),
                       "cudaMalloc d_pred_match");
        }
    }

    cudaEvent_t total_start{};
    cudaEvent_t total_stop{};
    cudaEvent_t h2d_start{};
    cudaEvent_t h2d_stop{};
    cudaEvent_t atomic_start{};
    cudaEvent_t atomic_stop{};
    cudaEvent_t d2h_start{};
    cudaEvent_t d2h_stop{};
    event_create(total_start);
    event_create(total_stop);
    event_create(h2d_start);
    event_create(h2d_stop);
    event_create(atomic_start);
    event_create(atomic_stop);
    event_create(d2h_start);
    event_create(d2h_stop);

    double acc_h2d = 0.0;
    double acc_groupby = 0.0;
    double acc_d2h = 0.0;
    double acc_total = 0.0;

    const int threads = threads_per_block;

    std::vector<double> partial_sums(num_batches * num_groups);
    std::vector<unsigned long long> partial_counts(num_batches * num_groups);
    double* h_partial_sums_pinned = nullptr;
    unsigned long long* h_partial_counts_pinned = nullptr;
    unsigned long long* h_batch_selected_pinned = nullptr;
    std::uint64_t selected_for_selectivity = 0;
    const char* iter_prefix = nvtx_iter_prefix_or_default(nvtx_iteration_prefix);

    if (multi_stream) {
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&h_partial_sums_pinned),
                                  num_batches * num_groups * sizeof(double)),
                   "cudaMallocHost partial_sums");
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&h_partial_counts_pinned),
                                  num_batches * num_groups * sizeof(unsigned long long)),
                   "cudaMallocHost partial_counts");
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&h_batch_selected_pinned),
                                  num_batches * sizeof(unsigned long long)),
                   "cudaMallocHost batch_selected");
    }

    CudaStreamGuard stream_guard(multi_stream ? ExecutionMode::Sync : execution);
    const cudaStream_t stream = stream_guard.stream();
    if (multi_stream) {
        NvtxRange execution_range("execution_multi_stream_batched");
    } else if (stream_guard.async()) {
        NvtxRange execution_range("execution_single_stream_async");
        // This path is async-capable but intentionally serialized.
        // Because H2D, kernels, and D2H are issued into the same stream,
        // they preserve order and are not expected to overlap.
        // Multi-stream batching will be introduced in a later step.
    } else {
        NvtxRange execution_range("execution_sync");
    }

    for (int it = 0; it < iterations; ++it) {
        NvtxRange iteration_range(iter_prefix, it);
        PipelineTiming iter_timing{};
        std::uint64_t iter_selected = 0;
        if (multi_stream) {
            std::fill(h_batch_selected_pinned, h_batch_selected_pinned + num_batches, 0ULL);
        }

        if (multi_stream) {
            NvtxRange batched_range("multi_stream_batched_pipeline");
            CpuTimer total_timer;
            for (std::size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
                const int stream_index = static_cast<int>(batch_id % static_cast<std::size_t>(active_streams));
                AtomicStreamContext& ctx = stream_contexts[static_cast<std::size_t>(stream_index)];
                std::string batch_label =
                    "batch_" + std::to_string(batch_id) + "_stream_" + std::to_string(stream_index);
                NvtxRange batch_range(batch_label.c_str());

                if (batch_id >= static_cast<std::size_t>(active_streams)) {
                    cuda_check(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize reuse");
                }

                const std::size_t batch_start = batch_id * effective_batch_rows;
                const std::size_t batch_count = std::min(effective_batch_rows, n - batch_start);
                const int blocks = static_cast<int>((batch_count + static_cast<std::size_t>(threads) - 1U) /
                                                    static_cast<std::size_t>(threads));

                memset_device(ctx.d_group_sum, 0, static_cast<std::size_t>(num_groups) * sizeof(double), ctx.stream);
                memset_device(ctx.d_group_count, 0,
                             static_cast<std::size_t>(num_groups) * sizeof(unsigned long long), ctx.stream);
                memset_device(ctx.d_pred_match, 0, sizeof(unsigned long long), ctx.stream);

                {
                    NvtxRange h2d_range("h2d_transfer");
                    memcpy_h2d(ctx.d_trip, columns.trip_distance.data() + batch_start, batch_count * sizeof(float),
                               ctx.stream);
                    memcpy_h2d(ctx.d_fare, columns.fare_amount.data() + batch_start, batch_count * sizeof(float),
                               ctx.stream);
                    memcpy_h2d(ctx.d_pc, columns.passenger_count.data() + batch_start,
                               batch_count * sizeof(std::uint32_t), ctx.stream);
                }

                {
                    NvtxRange kernel_range("atomic_groupby_sum_kernel_launch");
                    filter_groupby_atomic_naive_kernel<<<blocks, threads, 0, ctx.stream>>>(
                        ctx.d_trip, ctx.d_fare, ctx.d_pc, batch_count, ctx.d_group_sum, ctx.d_group_count,
                        ctx.d_pred_match, static_cast<std::uint32_t>(num_groups));
                    cuda_check(cudaGetLastError(), "filter_groupby_atomic_naive_kernel launch");
                }

                const std::size_t partial_offset = batch_id * num_groups;
                {
                    NvtxRange d2h_range("d2h_transfer");
                    memcpy_d2h(h_partial_sums_pinned + partial_offset, ctx.d_group_sum,
                               static_cast<std::size_t>(num_groups) * sizeof(double), ctx.stream);
                    memcpy_d2h(h_partial_counts_pinned + partial_offset, ctx.d_group_count,
                               static_cast<std::size_t>(num_groups) * sizeof(unsigned long long), ctx.stream);
                    memcpy_d2h(h_batch_selected_pinned + batch_id, ctx.d_pred_match, sizeof(unsigned long long),
                               ctx.stream);
                }
            }

            for (auto& ctx : stream_contexts) {
                cuda_check(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize final");
            }
            iter_timing.total_gpu_ms = total_timer.elapsed_ms();

            for (std::size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
                iter_selected += h_batch_selected_pinned[batch_id];
            }

            // Copy pinned partials into vectors used by CPU merge.
            std::copy(h_partial_sums_pinned, h_partial_sums_pinned + num_batches * num_groups, partial_sums.begin());
            std::copy(h_partial_counts_pinned, h_partial_counts_pinned + num_batches * num_groups,
                      partial_counts.begin());
        } else {
            NvtxRange batched_range("batched_pipeline");
            for (std::size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
                NvtxRange batch_range("batch", static_cast<int>(batch_id));
                const std::size_t batch_start = batch_id * effective_batch_rows;
                const std::size_t batch_count = std::min(effective_batch_rows, n - batch_start);
                const int blocks = static_cast<int>((batch_count + static_cast<std::size_t>(threads) - 1U) /
                                                    static_cast<std::size_t>(threads));

                memset_device(d_group_sum, 0, static_cast<std::size_t>(num_groups) * sizeof(double), stream);
                memset_device(d_group_count, 0, static_cast<std::size_t>(num_groups) * sizeof(unsigned long long),
                             stream);
                memset_device(d_pred_match, 0, sizeof(unsigned long long), stream);

                event_record(total_start, stream);

                event_record(h2d_start, stream);
                {
                    NvtxRange h2d_range("h2d_transfer");
                    memcpy_h2d(d_trip, columns.trip_distance.data() + batch_start, batch_count * sizeof(float),
                               stream);
                    memcpy_h2d(d_fare, columns.fare_amount.data() + batch_start, batch_count * sizeof(float), stream);
                    memcpy_h2d(d_pc, columns.passenger_count.data() + batch_start, batch_count * sizeof(std::uint32_t),
                               stream);
                }
                event_record(h2d_stop, stream);

                event_record(atomic_start, stream);
                {
                    NvtxRange kernel_range("atomic_groupby_sum_kernel_launch");
                    filter_groupby_atomic_naive_kernel<<<blocks, threads, 0, stream>>>(
                        d_trip, d_fare, d_pc, batch_count, d_group_sum, d_group_count, d_pred_match,
                        static_cast<std::uint32_t>(num_groups));
                    cuda_check(cudaGetLastError(), "filter_groupby_atomic_naive_kernel launch");
                }
                event_record(atomic_stop, stream);

                const std::size_t partial_offset = batch_id * num_groups;
                unsigned long long batch_selected = 0;
                event_record(d2h_start, stream);
                {
                    NvtxRange d2h_range("d2h_transfer");
                    memcpy_d2h(partial_sums.data() + partial_offset, d_group_sum,
                               static_cast<std::size_t>(num_groups) * sizeof(double), stream);
                    memcpy_d2h(partial_counts.data() + partial_offset, d_group_count,
                               static_cast<std::size_t>(num_groups) * sizeof(unsigned long long), stream);
                    memcpy_d2h(&batch_selected, d_pred_match, sizeof(unsigned long long), stream);
                }
                event_record(d2h_stop, stream);

                event_record(total_stop, stream);

                PipelineTiming batch_timing{};
                record_pipeline_timing_atomic(total_start, total_stop, h2d_start, h2d_stop, atomic_start, atomic_stop,
                                              d2h_start, d2h_stop, batch_timing, stream);
                add_pipeline_timing(iter_timing, batch_timing);
                iter_selected += batch_selected;
            }
        }

        {
            NvtxRange merge_range("cpu_merge_partial_results");
            CpuTimer merge_timer;
            merge_partial_groupby_cpu(partial_sums, partial_counts, num_batches, num_groups, last);
            iter_timing.cpu_merge_ms = merge_timer.elapsed_ms();
        }

        selected_for_selectivity = iter_selected;
        acc_h2d += iter_timing.h2d_ms;
        acc_groupby += iter_timing.atomic_groupby_kernel_ms;
        acc_d2h += iter_timing.d2h_ms;
        acc_total += iter_timing.total_gpu_ms;
        if (measured_timing != nullptr) {
            measured_timing->add(iter_timing);
        }
    }

    event_destroy(total_start);
    event_destroy(total_stop);
    event_destroy(h2d_start);
    event_destroy(h2d_stop);
    event_destroy(atomic_start);
    event_destroy(atomic_stop);
    event_destroy(d2h_start);
    event_destroy(d2h_stop);
    {
        NvtxRange free_range("cuda_free");
        if (multi_stream) {
            for (auto& ctx : stream_contexts) {
                ctx.free();
            }
            if (h_partial_sums_pinned != nullptr) {
                cudaFreeHost(h_partial_sums_pinned);
            }
            if (h_partial_counts_pinned != nullptr) {
                cudaFreeHost(h_partial_counts_pinned);
            }
            if (h_batch_selected_pinned != nullptr) {
                cudaFreeHost(h_batch_selected_pinned);
            }
        } else {
            cudaFree(d_trip);
            cudaFree(d_fare);
            cudaFree(d_pc);
            cudaFree(d_group_sum);
            cudaFree(d_group_count);
            cudaFree(d_pred_match);
        }
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
                                                         const char* nvtx_iteration_prefix, TimingStats* measured_timing,
                                                         ExecutionMode execution, std::size_t batch_rows,
                                                         int num_streams) {
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
    if (num_streams < 1) {
        throw std::runtime_error("num_streams must be >= 1");
    }

    const std::size_t n = columns.trip_distance.size();
    std::size_t effective_batch_rows = 0;
    std::size_t num_batches = 0;
    compute_batch_plan(n, batch_rows, effective_batch_rows, num_batches);
    const std::size_t max_batch_rows = effective_batch_rows;
    const std::size_t payload_bytes = n * (2U * sizeof(float) + sizeof(std::uint32_t));
    const bool multi_stream = (execution == ExecutionMode::MultiStreamBatched);
    const int active_streams = multi_stream ? num_streams : 1;

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
    std::vector<BlockPartialStreamContext> stream_contexts;

    {
        NvtxRange alloc_range("cuda_allocation");
        if (multi_stream) {
            stream_contexts.resize(static_cast<std::size_t>(active_streams));
            for (auto& ctx : stream_contexts) {
                ctx.allocate(max_batch_rows, num_groups);
            }
        } else {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_trip), max_batch_rows * sizeof(float)),
                       "cudaMalloc d_trip");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_fare), max_batch_rows * sizeof(float)),
                       "cudaMalloc d_fare");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pc), max_batch_rows * sizeof(std::uint32_t)),
                       "cudaMalloc d_pc");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask), max_batch_rows * sizeof(std::uint8_t)),
                       "cudaMalloc d_mask");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask_u64), max_batch_rows * sizeof(std::uint64_t)),
                       "cudaMalloc d_mask_u64");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_offsets), max_batch_rows * sizeof(std::uint64_t)),
                       "cudaMalloc d_offsets");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_selected_indices),
                                  max_batch_rows * sizeof(std::uint32_t)),
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
    }

    const int threads = threads_per_block;

    cudaEvent_t total_start{};
    cudaEvent_t total_stop{};
    cudaEvent_t h2d_start{};
    cudaEvent_t h2d_stop{};
    cudaEvent_t filter_start{};
    cudaEvent_t filter_stop{};
    cudaEvent_t scan_start{};
    cudaEvent_t scan_stop{};
    cudaEvent_t scatter_start{};
    cudaEvent_t scatter_stop{};
    cudaEvent_t reduce_start{};
    cudaEvent_t reduce_stop{};
    cudaEvent_t d2h_start{};
    cudaEvent_t d2h_stop{};
    event_create(total_start);
    event_create(total_stop);
    event_create(h2d_start);
    event_create(h2d_stop);
    event_create(filter_start);
    event_create(filter_stop);
    event_create(scan_start);
    event_create(scan_stop);
    event_create(scatter_start);
    event_create(scatter_stop);
    event_create(reduce_start);
    event_create(reduce_stop);
    event_create(d2h_start);
    event_create(d2h_stop);

    double acc_h2d = 0.0;
    double acc_filter = 0.0;
    double acc_scan = 0.0;
    double acc_scatter = 0.0;
    double acc_groupby = 0.0;
    double acc_d2h = 0.0;
    double acc_total = 0.0;

    std::vector<double> partial_sums(num_batches * num_groups);
    std::vector<unsigned long long> partial_counts(num_batches * num_groups);
    double* h_partial_sums_pinned = nullptr;
    unsigned long long* h_partial_counts_pinned = nullptr;
    unsigned long long* h_batch_selected_pinned = nullptr;
    unsigned long long last_selected = 0;
    const char* iter_prefix = nvtx_iter_prefix_or_default(nvtx_iteration_prefix);

    if (multi_stream) {
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&h_partial_sums_pinned),
                                  num_batches * num_groups * sizeof(double)),
                   "cudaMallocHost partial_sums");
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&h_partial_counts_pinned),
                                  num_batches * num_groups * sizeof(unsigned long long)),
                   "cudaMallocHost partial_counts");
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&h_batch_selected_pinned),
                                  num_batches * sizeof(unsigned long long)),
                   "cudaMallocHost batch_selected");
    }

    CudaStreamGuard stream_guard(multi_stream ? ExecutionMode::Sync : execution);
    const cudaStream_t stream = stream_guard.stream();
    if (multi_stream) {
        NvtxRange execution_range("execution_multi_stream_batched");
    } else if (stream_guard.async()) {
        NvtxRange execution_range("execution_single_stream_async");
        // This path is async-capable but intentionally serialized.
        // Because H2D, kernels, and D2H are issued into the same stream,
        // they preserve order and are not expected to overlap.
        // Multi-stream batching will be introduced in a later step.
    } else {
        NvtxRange execution_range("execution_sync");
    }

    for (int it = 0; it < iterations; ++it) {
        NvtxRange iteration_range(iter_prefix, it);
        PipelineTiming iter_timing{};
        unsigned long long iter_selected = 0;
        if (multi_stream) {
            std::fill(h_batch_selected_pinned, h_batch_selected_pinned + num_batches, 0ULL);
        }

        if (multi_stream) {
            NvtxRange batched_range("multi_stream_batched_pipeline");
            CpuTimer total_timer;
            for (std::size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
                const int stream_index = static_cast<int>(batch_id % static_cast<std::size_t>(active_streams));
                BlockPartialStreamContext& ctx = stream_contexts[static_cast<std::size_t>(stream_index)];
                std::string batch_label =
                    "batch_" + std::to_string(batch_id) + "_stream_" + std::to_string(stream_index);
                NvtxRange batch_range(batch_label.c_str());

                if (batch_id >= static_cast<std::size_t>(active_streams)) {
                    cuda_check(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize reuse");
                }

                const std::size_t batch_start = batch_id * effective_batch_rows;
                const std::size_t batch_count = std::min(effective_batch_rows, n - batch_start);
                const int blocks = static_cast<int>((batch_count + static_cast<std::size_t>(threads) - 1U) /
                                                    static_cast<std::size_t>(threads));
                const int reduce_blocks = compute_blocks(batch_count, threads);

                memset_device(ctx.d_selected_count, 0, sizeof(unsigned long long), ctx.stream);
                memset_device(ctx.d_group_sum, 0, static_cast<std::size_t>(num_groups) * sizeof(double), ctx.stream);
                memset_device(ctx.d_group_count, 0,
                             static_cast<std::size_t>(num_groups) * sizeof(unsigned long long), ctx.stream);

                {
                    NvtxRange h2d_range("h2d_transfer");
                    memcpy_h2d(ctx.d_trip, columns.trip_distance.data() + batch_start, batch_count * sizeof(float),
                               ctx.stream);
                    memcpy_h2d(ctx.d_fare, columns.fare_amount.data() + batch_start, batch_count * sizeof(float),
                               ctx.stream);
                    memcpy_h2d(ctx.d_pc, columns.passenger_count.data() + batch_start,
                               batch_count * sizeof(std::uint32_t), ctx.stream);
                }

                {
                    NvtxRange filter_range("filter_kernel_launch");
                    filter_mask_kernel<<<blocks, threads, 0, ctx.stream>>>(ctx.d_trip, ctx.d_fare, batch_count,
                                                                           ctx.d_mask);
                    cuda_check(cudaGetLastError(), "filter_mask_kernel launch");
                }

                {
                    NvtxRange scan_range("scan_kernel_launch");
                    mask_u8_to_u64_kernel<<<blocks, threads, 0, ctx.stream>>>(ctx.d_mask, ctx.d_mask_u64,
                                                                              batch_count);
                    cuda_check(cudaGetLastError(), "mask_u8_to_u64_kernel launch");
                    thrust::exclusive_scan(thrust::cuda::par.on(ctx.stream), ctx.d_mask_u64,
                                           ctx.d_mask_u64 + batch_count, ctx.d_offsets);
                    cuda_check(cudaGetLastError(), "thrust::exclusive_scan");
                }

                {
                    NvtxRange scatter_range("scatter_kernel_launch");
                    scatter_selected_indices_kernel<<<blocks, threads, 0, ctx.stream>>>(
                        ctx.d_mask_u64, ctx.d_offsets, batch_count, ctx.d_selected_indices);
                    cuda_check(cudaGetLastError(), "scatter_selected_indices_kernel launch");
                    write_selected_count_kernel<<<1, 1, 0, ctx.stream>>>(ctx.d_mask_u64, ctx.d_offsets, batch_count,
                                                                         ctx.d_selected_count);
                    cuda_check(cudaGetLastError(), "write_selected_count_kernel launch");
                }

                {
                    NvtxRange reduce_range("block_partial_reduce_kernel_launch");
                    const std::size_t shmem_bytes =
                        static_cast<std::size_t>(num_groups) * (sizeof(double) + sizeof(unsigned long long));
                    groupby_selected_block_partial_kernel<<<reduce_blocks, threads, shmem_bytes, ctx.stream>>>(
                        ctx.d_fare, ctx.d_pc, ctx.d_selected_indices, ctx.d_selected_count, ctx.d_group_sum,
                        ctx.d_group_count, static_cast<std::uint32_t>(num_groups));
                    cuda_check(cudaGetLastError(), "groupby_selected_block_partial_kernel launch");
                }

                const std::size_t partial_offset = batch_id * num_groups;
                {
                    NvtxRange d2h_range("d2h_transfer");
                    memcpy_d2h(h_batch_selected_pinned + batch_id, ctx.d_selected_count, sizeof(unsigned long long),
                               ctx.stream);
                    memcpy_d2h(h_partial_sums_pinned + partial_offset, ctx.d_group_sum,
                               static_cast<std::size_t>(num_groups) * sizeof(double), ctx.stream);
                    memcpy_d2h(h_partial_counts_pinned + partial_offset, ctx.d_group_count,
                               static_cast<std::size_t>(num_groups) * sizeof(unsigned long long), ctx.stream);
                }
            }

            for (auto& ctx : stream_contexts) {
                cuda_check(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize final");
            }
            iter_timing.total_gpu_ms = total_timer.elapsed_ms();

            for (std::size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
                iter_selected += h_batch_selected_pinned[batch_id];
            }
            std::copy(h_partial_sums_pinned, h_partial_sums_pinned + num_batches * num_groups, partial_sums.begin());
            std::copy(h_partial_counts_pinned, h_partial_counts_pinned + num_batches * num_groups,
                      partial_counts.begin());
        } else {
            NvtxRange batched_range("batched_pipeline");
            for (std::size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
                NvtxRange batch_range("batch", static_cast<int>(batch_id));
                const std::size_t batch_start = batch_id * effective_batch_rows;
                const std::size_t batch_count = std::min(effective_batch_rows, n - batch_start);
                const int blocks = static_cast<int>((batch_count + static_cast<std::size_t>(threads) - 1U) /
                                                    static_cast<std::size_t>(threads));
                const int reduce_blocks = compute_blocks(batch_count, threads);

                memset_device(d_selected_count, 0, sizeof(unsigned long long), stream);
                memset_device(d_group_sum, 0, static_cast<std::size_t>(num_groups) * sizeof(double), stream);
                memset_device(d_group_count, 0, static_cast<std::size_t>(num_groups) * sizeof(unsigned long long),
                             stream);

                event_record(total_start, stream);

                event_record(h2d_start, stream);
                {
                    NvtxRange h2d_range("h2d_transfer");
                    memcpy_h2d(d_trip, columns.trip_distance.data() + batch_start, batch_count * sizeof(float),
                               stream);
                    memcpy_h2d(d_fare, columns.fare_amount.data() + batch_start, batch_count * sizeof(float), stream);
                    memcpy_h2d(d_pc, columns.passenger_count.data() + batch_start, batch_count * sizeof(std::uint32_t),
                               stream);
                }
                event_record(h2d_stop, stream);

                event_record(filter_start, stream);
                {
                    NvtxRange filter_range("filter_kernel_launch");
                    filter_mask_kernel<<<blocks, threads, 0, stream>>>(d_trip, d_fare, batch_count, d_mask);
                    cuda_check(cudaGetLastError(), "filter_mask_kernel launch");
                }
                event_record(filter_stop, stream);

                event_record(scan_start, stream);
                {
                    NvtxRange scan_range("scan_kernel_launch");
                    mask_u8_to_u64_kernel<<<blocks, threads, 0, stream>>>(d_mask, d_mask_u64, batch_count);
                    cuda_check(cudaGetLastError(), "mask_u8_to_u64_kernel launch");
                    if (stream != nullptr) {
                        thrust::exclusive_scan(thrust::cuda::par.on(stream), d_mask_u64, d_mask_u64 + batch_count,
                                               d_offsets);
                    } else {
                        thrust::exclusive_scan(thrust::device, d_mask_u64, d_mask_u64 + batch_count, d_offsets);
                    }
                    cuda_check(cudaGetLastError(), "thrust::exclusive_scan");
                }
                event_record(scan_stop, stream);

                event_record(scatter_start, stream);
                {
                    NvtxRange scatter_range("scatter_kernel_launch");
                    scatter_selected_indices_kernel<<<blocks, threads, 0, stream>>>(d_mask_u64, d_offsets, batch_count,
                                                                                    d_selected_indices);
                    cuda_check(cudaGetLastError(), "scatter_selected_indices_kernel launch");
                    write_selected_count_kernel<<<1, 1, 0, stream>>>(d_mask_u64, d_offsets, batch_count,
                                                                     d_selected_count);
                    cuda_check(cudaGetLastError(), "write_selected_count_kernel launch");
                }
                event_record(scatter_stop, stream);

                event_record(reduce_start, stream);
                {
                    NvtxRange reduce_range("block_partial_reduce_kernel_launch");
                    const std::size_t shmem_bytes =
                        static_cast<std::size_t>(num_groups) * (sizeof(double) + sizeof(unsigned long long));
                    groupby_selected_block_partial_kernel<<<reduce_blocks, threads, shmem_bytes, stream>>>(
                        d_fare, d_pc, d_selected_indices, d_selected_count, d_group_sum, d_group_count,
                        static_cast<std::uint32_t>(num_groups));
                    cuda_check(cudaGetLastError(), "groupby_selected_block_partial_kernel launch");
                }
                event_record(reduce_stop, stream);

                const std::size_t partial_offset = batch_id * num_groups;
                unsigned long long batch_selected = 0;
                event_record(d2h_start, stream);
                {
                    NvtxRange d2h_range("d2h_transfer");
                    memcpy_d2h(&batch_selected, d_selected_count, sizeof(unsigned long long), stream);
                    memcpy_d2h(partial_sums.data() + partial_offset, d_group_sum,
                               static_cast<std::size_t>(num_groups) * sizeof(double), stream);
                    memcpy_d2h(partial_counts.data() + partial_offset, d_group_count,
                               static_cast<std::size_t>(num_groups) * sizeof(unsigned long long), stream);
                }
                event_record(d2h_stop, stream);

                event_record(total_stop, stream);

                PipelineTiming batch_timing{};
                record_pipeline_timing_block_partial(total_start, total_stop, h2d_start, h2d_stop, filter_start,
                                                     filter_stop, scan_start, scan_stop, scatter_start, scatter_stop,
                                                     reduce_start, reduce_stop, d2h_start, d2h_stop, batch_timing,
                                                     stream);
                add_pipeline_timing(iter_timing, batch_timing);
                iter_selected += batch_selected;
            }
        }

        {
            NvtxRange merge_range("cpu_merge_partial_results");
            CpuTimer merge_timer;
            merge_partial_groupby_cpu(partial_sums, partial_counts, num_batches, num_groups, last);
            iter_timing.cpu_merge_ms = merge_timer.elapsed_ms();
        }

        last_selected = iter_selected;
        acc_h2d += iter_timing.h2d_ms;
        acc_filter += iter_timing.filter_kernel_ms;
        acc_scan += iter_timing.scan_kernel_ms;
        acc_scatter += iter_timing.scatter_kernel_ms;
        acc_groupby += iter_timing.block_partial_reduce_kernel_ms;
        acc_d2h += iter_timing.d2h_ms;
        acc_total += iter_timing.total_gpu_ms;
        if (measured_timing != nullptr) {
            measured_timing->add(iter_timing);
        }
    }

    event_destroy(total_start);
    event_destroy(total_stop);
    event_destroy(h2d_start);
    event_destroy(h2d_stop);
    event_destroy(filter_start);
    event_destroy(filter_stop);
    event_destroy(scan_start);
    event_destroy(scan_stop);
    event_destroy(scatter_start);
    event_destroy(scatter_stop);
    event_destroy(reduce_start);
    event_destroy(reduce_stop);
    event_destroy(d2h_start);
    event_destroy(d2h_stop);
    {
        NvtxRange free_range("cuda_free");
        if (multi_stream) {
            for (auto& ctx : stream_contexts) {
                ctx.free();
            }
            if (h_partial_sums_pinned != nullptr) {
                cudaFreeHost(h_partial_sums_pinned);
            }
            if (h_partial_counts_pinned != nullptr) {
                cudaFreeHost(h_partial_counts_pinned);
            }
            if (h_batch_selected_pinned != nullptr) {
                cudaFreeHost(h_batch_selected_pinned);
            }
        } else {
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
