
#include "execution/gpu_executor.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

bool report_cuda_error(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed: "
              << cudaGetErrorString(error) << '\n';
    return false;
}

// CUDA events measure work on the device timeline.
// Event creation/destruction is not included in event timings.
class CudaEventPair {
public:
    CudaEventPair() = default;

    ~CudaEventPair()
    {
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }

        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
    }

    CudaEventPair(const CudaEventPair&) = delete;
    CudaEventPair& operator=(const CudaEventPair&) = delete;

    bool initialize()
    {
        if (!report_cuda_error(
                cudaEventCreate(&start_),
                "cudaEventCreate(start)")) {
            return false;
        }

        if (!report_cuda_error(
                cudaEventCreate(&stop_),
                "cudaEventCreate(stop)")) {
            return false;
        }

        return true;
    }

    bool start()
    {
        return report_cuda_error(
            cudaEventRecord(start_),
            "cudaEventRecord(start)");
    }

    bool stop_and_measure(double& milliseconds)
    {
        if (!report_cuda_error(
                cudaEventRecord(stop_),
                "cudaEventRecord(stop)")) {
            return false;
        }

        if (!report_cuda_error(
                cudaEventSynchronize(stop_),
                "cudaEventSynchronize") ) {
            return false;
        }

        float measured_ms = 0.0F;

        if (!report_cuda_error(
                cudaEventElapsedTime(
                    &measured_ms,
                    start_,
                    stop_),
                "cudaEventElapsedTime")) {
            return false;
        }

        milliseconds = static_cast<double>(measured_ms);
        return true;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

// Owns device allocations and releases them on every exit path.
class DeviceBuffers {
public:
    DeviceBuffers() = default;

    ~DeviceBuffers()
    {
        release();
    }

    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;

    bool allocate(
        std::size_t bytes_a,
        std::size_t bytes_b,
        std::size_t bytes_c)
    {
        if (!report_cuda_error(
                cudaMalloc(&a_, bytes_a),
                "cudaMalloc(A)")) {
            return false;
        }

        if (!report_cuda_error(
                cudaMalloc(&b_, bytes_b),
                "cudaMalloc(B)")) {
            return false;
        }

        if (!report_cuda_error(
                cudaMalloc(&c_, bytes_c),
                "cudaMalloc(C)")) {
            return false;
        }

        return true;
    }

    float* a() const { return a_; }
    float* b() const { return b_; }
    float* c() const { return c_; }

    // Returns host time spent requesting device-buffer release.
    // Any outstanding CUDA work is synchronized before this is called.
    double release_timed()
    {
        const auto start = Clock::now();
        release();
        const auto end = Clock::now();

        return elapsed_ms(start, end);
    }

private:
    void release() noexcept
    {
        if (c_ != nullptr) {
            cudaFree(c_);
            c_ = nullptr;
        }

        if (b_ != nullptr) {
            cudaFree(b_);
            b_ = nullptr;
        }

        if (a_ != nullptr) {
            cudaFree(a_);
            a_ = nullptr;
        }
    }

    float* a_ = nullptr;
    float* b_ = nullptr;
    float* c_ = nullptr;
};

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    std::size_t n)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

__global__ void matrix_multiply_kernel(
    const float* a,
    const float* b,
    float* c,
    std::size_t n)
{
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y
        + threadIdx.y;

    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0F;

    for (std::size_t k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }

    c[row * n + col] = sum;
}

bool valid_vector_inputs(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c,
    std::size_t n)
{
    return n > 0
        && a.size() == n
        && b.size() == n
        && c.size() == n
        && n <= static_cast<std::size_t>(
            std::numeric_limits<int>::max()) * 256ULL;
}

bool valid_matrix_inputs(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c,
    std::size_t n)
{
    if (n == 0
        || n > std::numeric_limits<std::size_t>::max() / n) {
        return false;
    }

    const std::size_t elements = n * n;

    return a.size() == elements
        && b.size() == elements
        && c.size() == elements;
}

} // namespace

GPUExecutor::~GPUExecutor()
{
    shutdown();
}

bool GPUExecutor::initialize()
{
    if (initialized) {
        return true;
    }

    int device_count = 0;

    if (!report_cuda_error(
            cudaGetDeviceCount(&device_count),
            "cudaGetDeviceCount")) {
        return false;
    }

    if (device_count <= 0) {
        std::cerr << "No CUDA-capable GPU detected.\n";
        return false;
    }

    if (!report_cuda_error(
            cudaSetDevice(0),
            "cudaSetDevice")) {
        return false;
    }

    cudaDeviceProp properties{};

    if (!report_cuda_error(
            cudaGetDeviceProperties(&properties, 0),
            "cudaGetDeviceProperties")) {
        return false;
    }

    std::cout << "CUDA GPU: " << properties.name << '\n';

    initialized = true;
    return true;
}

GPUExecutionResult GPUExecutor::vector_add(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::size_t N)
{
    GPUExecutionResult result{};
    const auto total_start = Clock::now();

    auto finish = [&]() {
        result.total_ms = elapsed_ms(total_start, Clock::now());
        return result;
    };

    if (!initialized) {
        std::cerr << "GPU executor is not initialized.\n";
        return finish();
    }

    if (!valid_vector_inputs(A, B, C, N)) {
        std::cerr << "Invalid vector-add input sizes.\n";
        return finish();
    }

    if (N > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        std::cerr << "Vector size overflows byte count.\n";
        return finish();
    }

    const std::size_t bytes = N * sizeof(float);

    DeviceBuffers buffers;

    CudaEventPair events;

    if (!events.initialize()) {
        return finish();
    }

    const auto allocation_start = Clock::now();

    const bool allocated = buffers.allocate(bytes, bytes, bytes);

    result.allocation_ms =
        elapsed_ms(allocation_start, Clock::now());

    if (!allocated) {
        result.cleanup_ms = buffers.release_timed();
        return finish();
    }

    bool success = true;

    if (events.start()) {
        const cudaError_t error =
            cudaMemcpy(
                buffers.a(), A.data(), bytes,
                cudaMemcpyHostToDevice);

        if (report_cuda_error(error, "cudaMemcpy(A to device)")) {
            success = events.stop_and_measure(result.host_to_device_ms);
        } else {
            success = false;
        }
    } else {
        success = false;
    }

    if (success && events.start()) {
        const cudaError_t error =
            cudaMemcpy(
                buffers.b(), B.data(), bytes,
                cudaMemcpyHostToDevice);

        double second_transfer_ms = 0.0;

        if (report_cuda_error(error, "cudaMemcpy(B to device)")) {
            success = events.stop_and_measure(second_transfer_ms);
            result.host_to_device_ms += second_transfer_ms;
        } else {
            success = false;
        }
    } else {
        success = false;
    }

    if (success && events.start()) {
        constexpr unsigned int threads = 256;

        const unsigned int blocks =
            static_cast<unsigned int>(
                (N + threads - 1) / threads);

        vector_add_kernel<<<blocks, threads>>>(
            buffers.a(), buffers.b(), buffers.c(), N);

        success = report_cuda_error(
            cudaGetLastError(),
            "vector_add_kernel launch");

        if (success) {
            success = events.stop_and_measure(result.kernel_ms);
        }
    } else {
        success = false;
    }

    if (success && events.start()) {
        const cudaError_t error =
            cudaMemcpy(
                C.data(), buffers.c(), bytes,
                cudaMemcpyDeviceToHost);

        if (report_cuda_error(error, "cudaMemcpy(C to host)")) {
            success = events.stop_and_measure(result.device_to_host_ms);
        } else {
            success = false;
        }
    } else {
        success = false;
    }

    // Ensure pending work has completed before releasing device memory.
    if (!report_cuda_error(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize")) {
        success = false;
    }

    result.cleanup_ms = buffers.release_timed();
    result.success = success;

    return finish();
}

GPUExecutionResult GPUExecutor::matrix_multiply(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::size_t N)
{
    GPUExecutionResult result{};
    const auto total_start = Clock::now();

    auto finish = [&]() {
        result.total_ms = elapsed_ms(total_start, Clock::now());
        return result;
    };

    if (!initialized) {
        std::cerr << "GPU executor is not initialized.\n";
        return finish();
    }

    if (!valid_matrix_inputs(A, B, C, N)) {
        std::cerr << "Invalid matrix-multiply input sizes.\n";
        return finish();
    }

    const std::size_t elements = N * N;

    if (elements > std::numeric_limits<std::size_t>::max()
                       / sizeof(float)) {
        std::cerr << "Matrix size overflows byte count.\n";
        return finish();
    }

    const std::size_t bytes = elements * sizeof(float);

    DeviceBuffers buffers;

    CudaEventPair events;

    if (!events.initialize()) {
        return finish();
    }

    const auto allocation_start = Clock::now();

    const bool allocated = buffers.allocate(bytes, bytes, bytes);

    result.allocation_ms =
        elapsed_ms(allocation_start, Clock::now());

    if (!allocated) {
        result.cleanup_ms = buffers.release_timed();
        return finish();
    }

    bool success = true;

    // Transfer A.
    if (events.start()) {
        const cudaError_t error =
            cudaMemcpy(
                buffers.a(), A.data(), bytes,
                cudaMemcpyHostToDevice);

        if (report_cuda_error(error, "cudaMemcpy(A to device)")) {
            success = events.stop_and_measure(result.host_to_device_ms);
        } else {
            success = false;
        }
    } else {
        success = false;
    }

    // Transfer B and add its timing to the first transfer.
    if (success && events.start()) {
        const cudaError_t error =
            cudaMemcpy(
                buffers.b(), B.data(), bytes,
                cudaMemcpyHostToDevice);

        double second_transfer_ms = 0.0;

        if (report_cuda_error(error, "cudaMemcpy(B to device)")) {
            success = events.stop_and_measure(second_transfer_ms);
            result.host_to_device_ms += second_transfer_ms;
        } else {
            success = false;
        }
    } else {
        success = false;
    }

    if (success && events.start()) {
        const dim3 threads(16, 16);
        const dim3 blocks(
            static_cast<unsigned int>((N + threads.x - 1) / threads.x),
            static_cast<unsigned int>((N + threads.y - 1) / threads.y));

        matrix_multiply_kernel<<<blocks, threads>>>(
            buffers.a(), buffers.b(), buffers.c(), N);

        success = report_cuda_error(
            cudaGetLastError(),
            "matrix_multiply_kernel launch");

        if (success) {
            success = events.stop_and_measure(result.kernel_ms);
        }
    } else {
        success = false;
    }

    if (success && events.start()) {
        const cudaError_t error =
            cudaMemcpy(
                C.data(), buffers.c(), bytes,
                cudaMemcpyDeviceToHost);

        if (report_cuda_error(error, "cudaMemcpy(C to host)")) {
            success = events.stop_and_measure(result.device_to_host_ms);
        } else {
            success = false;
        }
    } else {
        success = false;
    }

    if (!report_cuda_error(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize")) {
        success = false;
    }

    result.cleanup_ms = buffers.release_timed();
    result.success = success;

    return finish();
}

void GPUExecutor::shutdown()
{
    if (!initialized) {
        return;
    }

    // Do not reset the device: other components may use the CUDA context.
    initialized = false;
}
