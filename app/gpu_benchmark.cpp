#include "execution/gpu_executor.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {

    constexpr std::size_t N = 1024;

    std::cout
        << "========================================\n"
        << "      HeteroOS GPU Benchmark\n"
        << "========================================\n\n";

    std::cout
        << "Matrix size: "
        << N
        << " x "
        << N
        << "\n\n";

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C;

    for (std::size_t i = 0; i < N * N; ++i) {

        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    GPUExecutor executor;

    if (!executor.initialize()) {

        std::cerr
            << "Failed to initialize GPU executor\n";

        return 1;
    }

    // ----------------------------------
    // WARM-UP
    // ----------------------------------

    std::cout << "Warming up CUDA...\n";

    GPUExecutionResult warmup =
        executor.matrix_multiply(
            A,
            B,
            C,
            N
        );

    if (!warmup.correct) {

        std::cerr
            << "CUDA warm-up failed\n";

        return 1;
    }

    // ----------------------------------
    // BENCHMARK
    // ----------------------------------

    std::cout << "Running benchmark...\n\n";

    GPUExecutionResult result =
        executor.matrix_multiply(
            A,
            B,
            C,
            N
        );

    std::cout
        << std::fixed
        << std::setprecision(4);

    std::cout
        << "Host -> Device : "
        << result.host_to_device_ms
        << " ms\n";

    std::cout
        << "Kernel         : "
        << result.kernel_ms
        << " ms\n";

    std::cout
        << "Device -> Host : "
        << result.device_to_host_ms
        << " ms\n";

    std::cout
        << "Total GPU path : "
        << result.total_ms
        << " ms\n";

    std::cout
        << "Result         : "
        << (result.correct
            ? "CORRECT"
            : "INCORRECT")
        << '\n';

    return result.correct ? 0 : 1;
}