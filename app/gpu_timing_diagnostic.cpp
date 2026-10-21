
#include "execution/gpu_executor.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    constexpr std::size_t N = 256;
    constexpr int WARMUP_RUNS = 3;
    constexpr int MEASURED_RUNS = 20;

    const std::size_t elements = N * N;

    std::vector<float> A(elements, 1.0f);
    std::vector<float> B(elements, 2.0f);
    std::vector<float> C(elements, 0.0f);

    GPUExecutor gpu;

    if (!gpu.initialize()) {
        std::cerr << "GPU initialization failed.\n";
        return 1;
    }

    std::cout << "\nGPU Timing Diagnostic\n"
              << "Matrix dimension: " << N << '\n'
              << "Warm-ups: " << WARMUP_RUNS << '\n'
              << "Measured runs: " << MEASURED_RUNS << "\n\n";

    auto run_once = [&](int run, bool warmup) -> bool {
        const GPUExecutionResult result =
            gpu.matrix_multiply(A, B, C, N);

        if (!result.success) {
            std::cerr << "GPU execution failed at run "
                      << run << '\n';
            return false;
        }

        // Independent correctness check outside executor timing.
        for (float value : C) {
            if (std::fabs(value - static_cast<float>(2 * N)) > 1e-3f) {
                std::cerr << "Incorrect output at run " << run << '\n';
                return false;
            }
        }

        if (!warmup) {
            std::cout << std::setw(3) << run
                      << std::setw(14) << result.total_ms
                      << std::setw(14) << result.allocation_ms
                      << std::setw(14) << result.host_to_device_ms
                      << std::setw(14) << result.kernel_ms
                      << std::setw(14) << result.device_to_host_ms
                      << std::setw(14) << result.cleanup_ms
                      << '\n';
        }

        return true;
    };

    std::cout << "Running warm-ups...\n";

    for (int i = 1; i <= WARMUP_RUNS; ++i) {
        if (!run_once(i, true)) {
            gpu.shutdown();
            return 1;
        }
    }

    std::cout << "\n"
              << std::setw(3) << "Run"
              << std::setw(14) << "Total(ms)"
              << std::setw(14) << "Alloc(ms)"
              << std::setw(14) << "H2D(ms)"
              << std::setw(14) << "Kernel(ms)"
              << std::setw(14) << "D2H(ms)"
              << std::setw(14) << "Cleanup(ms)"
              << '\n';

    std::cout << std::fixed << std::setprecision(4);

    for (int i = 1; i <= MEASURED_RUNS; ++i) {
        if (!run_once(i, false)) {
            gpu.shutdown();
            return 1;
        }
    }

    gpu.shutdown();
    return 0;
}
