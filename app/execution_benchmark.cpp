#include "execution/cpu_executor.hpp"
#include "execution/gpu_executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

constexpr int MEASUREMENTS = 3;

struct CPUResult {
    double time_ms = 0.0;
    bool correct = false;
};

struct GPUMedianResult {
    double host_to_device_ms = 0.0;
    double kernel_ms = 0.0;
    double device_to_host_ms = 0.0;
    double total_ms = 0.0;
    bool correct = false;
};

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());

    const std::size_t n = values.size();

    if (n % 2 == 0) {
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    }

    return values[n / 2];
}

bool verify_result(
    const std::vector<float>& C,
    std::size_t N
)
{
    const float expected = 2.0f * static_cast<float>(N);

    for (float value : C) {
        if (std::fabs(value - expected) > 1e-3f) {
            return false;
        }
    }

    return true;
}

CPUResult benchmark_cpu(
    CPUExecutor& executor,
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::size_t N
)
{
    // Warm-up
    if (!executor.matrix_multiply(A, B, C, N)) {
        return {0.0, false};
    }

    if (!verify_result(C, N)) {
        return {0.0, false};
    }

    std::vector<double> timings;
    timings.reserve(MEASUREMENTS);

    for (int i = 0; i < MEASUREMENTS; ++i) {
        std::fill(C.begin(), C.end(), 0.0f);

        auto start = Clock::now();

        bool success =
            executor.matrix_multiply(A, B, C, N);

        auto end = Clock::now();

        if (!success || !verify_result(C, N)) {
            return {0.0, false};
        }

        double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        timings.push_back(elapsed_ms);
    }

    return {
        median(timings),
        true
    };
}

GPUMedianResult benchmark_gpu(
    GPUExecutor& executor,
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::size_t N
)
{
    // Warm-up
    GPUExecutionResult warmup =
        executor.matrix_multiply(A, B, C, N);

    if (!warmup.correct) {
        return {};
    }

    std::vector<double> h2d;
    std::vector<double> kernel;
    std::vector<double> d2h;
    std::vector<double> total;

    h2d.reserve(MEASUREMENTS);
    kernel.reserve(MEASUREMENTS);
    d2h.reserve(MEASUREMENTS);
    total.reserve(MEASUREMENTS);

    for (int i = 0; i < MEASUREMENTS; ++i) {
        std::fill(C.begin(), C.end(), 0.0f);

        GPUExecutionResult result =
            executor.matrix_multiply(A, B, C, N);

        if (!result.correct || !verify_result(C, N)) {
            return {};
        }

        h2d.push_back(result.host_to_device_ms);
        kernel.push_back(result.kernel_ms);
        d2h.push_back(result.device_to_host_ms);
        total.push_back(result.total_ms);
    }

    return {
        median(h2d),
        median(kernel),
        median(d2h),
        median(total),
        true
    };
}

int main()
{
    constexpr std::size_t sizes[] = {
        256,
        512,
        1024,
        1536
    };

    std::cout << "============================================\n";
    std::cout << "       HeteroOS Execution Benchmark\n";
    std::cout << "============================================\n\n";

    std::cout << "Measurements per workload: "
              << MEASUREMENTS << '\n';
    std::cout << "Statistic: median\n\n";

    CPUExecutor cpu;
    GPUExecutor gpu;

    if (!gpu.initialize()) {
        std::cerr << "Failed to initialize GPU executor.\n";
        return 1;
    }

    std::cout << "CUDA GPU: NVIDIA GeForce RTX 3050 A Laptop GPU\n\n";

    std::cout << std::left
              << std::setw(8)  << "N"
              << std::setw(14) << "CPU (ms)"
              << std::setw(14) << "GPU H2D"
              << std::setw(14) << "GPU Kernel"
              << std::setw(14) << "GPU D2H"
              << std::setw(14) << "GPU Total"
              << std::setw(12) << "Speedup"
              << '\n';

    std::cout << std::string(100, '-') << '\n';

    for (std::size_t N : sizes) {

        const std::size_t elements = N * N;

        std::vector<float> A(elements, 1.0f);
        std::vector<float> B(elements, 2.0f);

        std::vector<float> C_cpu(elements, 0.0f);
        std::vector<float> C_gpu(elements, 0.0f);

        // ------------------------------
        // CPU
        // ------------------------------

        CPUResult cpu_result =
            benchmark_cpu(
                cpu,
                A,
                B,
                C_cpu,
                N
            );

        if (!cpu_result.correct) {
            std::cerr
                << "CPU benchmark failed for N="
                << N << '\n';

            gpu.shutdown();
            return 1;
        }

        // ------------------------------
        // GPU
        // ------------------------------

        GPUMedianResult gpu_result =
            benchmark_gpu(
                gpu,
                A,
                B,
                C_gpu,
                N
            );

        if (!gpu_result.correct) {
            std::cerr
                << "GPU benchmark failed for N="
                << N << '\n';

            gpu.shutdown();
            return 1;
        }

        double speedup =
            cpu_result.time_ms /
            gpu_result.total_ms;

        std::cout << std::left
                  << std::setw(8)
                  << N

                  << std::setw(14)
                  << std::fixed
                  << std::setprecision(3)
                  << cpu_result.time_ms

                  << std::setw(14)
                  << gpu_result.host_to_device_ms

                  << std::setw(14)
                  << gpu_result.kernel_ms

                  << std::setw(14)
                  << gpu_result.device_to_host_ms

                  << std::setw(14)
                  << gpu_result.total_ms

                  << std::setw(12)
                  << speedup
                  << "x\n";
    }

    gpu.shutdown();

    std::cout << "\nBenchmark complete.\n";

    return 0;
}