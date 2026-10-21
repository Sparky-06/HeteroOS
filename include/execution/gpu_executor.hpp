#ifndef GPU_EXECUTOR_HPP
#define GPU_EXECUTOR_HPP

#include <cstddef>
#include <vector>


struct GPUExecutionResult {
    double allocation_ms = 0.0;
    double host_to_device_ms = 0.0;
    double kernel_ms = 0.0;
    double device_to_host_ms = 0.0;
    double cleanup_ms = 0.0;

    // Full wall-clock duration of the executor call.
    double total_ms = 0.0;

    bool success = false;
    bool correct = false;
};


class GPUExecutor {

public:

    GPUExecutor() = default;

    ~GPUExecutor();

    GPUExecutor(const GPUExecutor&) = delete;
    GPUExecutor& operator=(const GPUExecutor&) = delete;

    bool initialize();

    GPUExecutionResult matrix_multiply(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        std::size_t N
    );

    GPUExecutionResult vector_add(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        std::size_t N
    );

    void shutdown();

private:

    bool initialized = false;
};

#endif