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

constexpr int MEASUREMENTS = 5;

constexpr std::size_t VECTOR_SIZES[] = {
    1024,
    65536,
    1048576,
    4194304
};

constexpr std::size_t THREAD_COUNTS[] = {
    1, 2, 4, 6, 8, 10, 12
};

struct CPUResult {
    double latency_ms = 0.0;
    bool correct = false;
};

struct GPUResult {
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double transfer_kernel_ms = 0.0;
    double call_latency_ms = 0.0;
    bool correct = false;
};

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());

    const std::size_t n = values.size();

    if (n == 0) {
        return 0.0;
    }

    if (n % 2 == 0) {
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    }

    return values[n / 2];
}

bool verify_result(
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C
)
{
    if (A.size() != B.size() || A.size() != C.size()) {
        return false;
    }

    for (std::size_t i = 0; i < C.size(); ++i) {
        const float expected = A[i] + B[i];

        if (!std::isfinite(C[i]) ||
            std::fabs(C[i] - expected) > 1e-5f) {
            return false;
        }
    }

    return true;
}

int batch_size_for(std::size_t N)
{
    if (N < 4096) {
        return 1000;
    }

    if (N < 65536) {
        return 100;
    }

    if (N < 1048576) {
        return 10;
    }

    return 1;
}

// --------------------------------------------------
// CPU BENCHMARK
//
// Includes operation and correctness validation in
// each timed call. Tiny workloads are batched to
// reduce timer overhead.
// --------------------------------------------------

CPUResult benchmark_cpu(
    CPUExecutor& cpu,
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::size_t thread_count
)
{
    const std::size_t N = A.size();

    auto execute = [&]() {
        if (thread_count == 1) {
            return cpu.vector_add(A, B, C, N);
        }

        return cpu.vector_add_parallel(
            A, B, C, N, thread_count
        );
    };

    // Warm-up.
    if (!execute() || !verify_result(A, B, C)) {
        return {0.0, false};
    }

    const int batch_size = batch_size_for(N);

    std::vector<double> timings;
    timings.reserve(MEASUREMENTS);

    for (int sample = 0; sample < MEASUREMENTS; ++sample) {
        const auto start = Clock::now();

        for (int repetition = 0;
             repetition < batch_size;
             ++repetition) {

            if (!execute() || !verify_result(A, B, C)) {
                return {0.0, false};
            }
        }

        const auto end = Clock::now();

        // Independent post-sample correctness check.
        if (!verify_result(A, B, C)) {
            return {0.0, false};
        }

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        timings.push_back(elapsed_ms / batch_size);
    }

    return {median(timings), true};
}

// --------------------------------------------------
// GPU BENCHMARK
//
// call_latency_ms includes the complete vector_add()
// call, including device allocations, transfers,
// kernel execution, cleanup, and its internal
// correctness verification.
// --------------------------------------------------

GPUResult benchmark_gpu(
    GPUExecutor& gpu,
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C
)
{
    const std::size_t N = A.size();

    GPUExecutionResult warmup =
        gpu.vector_add(A, B, C, N);

    if (!warmup.correct || !verify_result(A, B, C)) {
        return {};
    }

    std::vector<double> h2d;
    std::vector<double> kernel;
    std::vector<double> d2h;
    std::vector<double> transfer_kernel;
    std::vector<double> call_latency;

    h2d.reserve(MEASUREMENTS);
    kernel.reserve(MEASUREMENTS);
    d2h.reserve(MEASUREMENTS);
    transfer_kernel.reserve(MEASUREMENTS);
    call_latency.reserve(MEASUREMENTS);

    for (int sample = 0; sample < MEASUREMENTS; ++sample) {
        const auto start = Clock::now();

        GPUExecutionResult result =
            gpu.vector_add(A, B, C, N);

        const auto end = Clock::now();

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        if (!result.correct || !verify_result(A, B, C)) {
            return {};
        }

        h2d.push_back(result.host_to_device_ms);
        kernel.push_back(result.kernel_ms);
        d2h.push_back(result.device_to_host_ms);
        transfer_kernel.push_back(result.total_ms);
        call_latency.push_back(elapsed_ms);
    }

    return {
        median(h2d),
        median(kernel),
        median(d2h),
        median(transfer_kernel),
        median(call_latency),
        true
    };
}

// --------------------------------------------------
// MAIN
// --------------------------------------------------

int main()
{
    CPUExecutor cpu;
    GPUExecutor gpu;

    if (!gpu.initialize()) {
        std::cerr << "GPU initialization failed.\n";
        return 1;
    }

    std::cout
        << "\n============================================\n"
        << " HeteroOS Final Vector Addition Benchmark\n"
        << "============================================\n"
        << "Statistic: median of "
        << MEASUREMENTS << " samples\n"
        << "CPU timings include correctness validation.\n"
        << "GPU call timings include executor overhead and validation.\n";

    for (std::size_t N : VECTOR_SIZES) {
        std::vector<float> A(N);
        std::vector<float> B(N);
        std::vector<float> C_cpu(N, 0.0f);
        std::vector<float> C_gpu(N, 0.0f);

        // Deterministic, non-constant input data.
        for (std::size_t i = 0; i < N; ++i) {
            A[i] = static_cast<float>(i % 101) * 0.25f;
            B[i] = static_cast<float>(i % 37) * 0.5f;
        }

        std::cout
            << "\n--------------------------------------------\n"
            << "Vector elements: " << N << '\n'
            << "--------------------------------------------\n";

        std::cout << std::left
                  << std::setw(10) << "Threads"
                  << std::setw(16) << "Latency (ms)"
                  << std::setw(12) << "Speedup"
                  << std::setw(14) << "Efficiency"
                  << '\n';

        std::cout << std::string(52, '-') << '\n';

        double baseline_ms = 0.0;

        for (std::size_t threads : THREAD_COUNTS) {
            CPUResult result = benchmark_cpu(
                cpu, A, B, C_cpu, threads
            );

            if (!result.correct || result.latency_ms <= 0.0) {
                std::cerr
                    << "CPU benchmark failed at N=" << N
                    << ", threads=" << threads << '\n';

                gpu.shutdown();
                return 1;
            }

            if (threads == 1) {
                baseline_ms = result.latency_ms;
            }

            const double speedup =
                baseline_ms / result.latency_ms;

            const double efficiency =
                speedup / static_cast<double>(threads) * 100.0;

            std::cout << std::left
                      << std::setw(10) << threads
                      << std::fixed << std::setprecision(5)
                      << std::setw(16) << result.latency_ms
                      << std::setprecision(3)
                      << std::setw(12) << speedup
                      << std::setprecision(2)
                      << std::setw(14) << efficiency
                      << "%\n";
        }

        GPUResult gpu_result =
            benchmark_gpu(gpu, A, B, C_gpu);

        if (!gpu_result.correct ||
            gpu_result.call_latency_ms <= 0.0) {

            std::cerr
                << "GPU benchmark failed at N=" << N << '\n';

            gpu.shutdown();
            return 1;
        }

        std::cout << std::fixed << std::setprecision(5);

        std::cout << "\nGPU median measurements:\n"
                  << "  H2D:                  "
                  << gpu_result.h2d_ms << " ms\n"
                  << "  Kernel:               "
                  << gpu_result.kernel_ms << " ms\n"
                  << "  D2H:                  "
                  << gpu_result.d2h_ms << " ms\n"
                  << "  Transfer + kernel:    "
                  << gpu_result.transfer_kernel_ms << " ms\n"
                  << "  Full GPU call:        "
                  << gpu_result.call_latency_ms << " ms\n"
                  << "  Full-call speedup vs CPU (1 thread): "
                  << std::setprecision(3)
                  << baseline_ms / gpu_result.call_latency_ms
                  << "x\n";
    }

    gpu.shutdown();

    std::cout
        << "\nBenchmark complete. All results verified.\n";

    return 0;
}