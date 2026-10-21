
#include "execution/gpu_executor.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

int main()
{
    constexpr std::size_t N = 256;
    constexpr int REPEATS = 10;
    const std::size_t elements = N * N;

    std::vector<float> A(elements, 1.0f);
    std::vector<float> B(elements, 2.0f);
    std::vector<float> C(elements, 0.0f);

    GPUExecutor gpu;

    if (!gpu.initialize()) {
        std::cerr << "GPU initialization failed.\n";
        return 1;
    }

    std::cout << "Waiting one second before first GPU call...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "\n"
              << std::setw(5) << "Run"
              << std::setw(15) << "Total (ms)"
              << std::setw(15) << "Kernel (ms)"
              << std::setw(15) << "Allocation"
              << '\n';

    std::cout << std::fixed << std::setprecision(4);

    for (int i = 1; i <= REPEATS + 1; ++i) {
        const GPUExecutionResult result =
            gpu.matrix_multiply(A, B, C, N);

        if (!result.success) {
            std::cerr << "GPU execution failed on run " << i << '\n';
            gpu.shutdown();
            return 1;
        }

        std::cout << std::setw(5) << i
                  << std::setw(15) << result.total_ms
                  << std::setw(15) << result.kernel_ms
                  << std::setw(15) << result.allocation_ms
                  << '\n';
    }

    gpu.shutdown();
    return 0;
}
