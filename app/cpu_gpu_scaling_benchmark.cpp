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

constexpr std::size_t N = 1024;
constexpr int MEASUREMENTS = 3;

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

double benchmark_cpu(
    CPUExecutor& executor,
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::size_t N,
    std::size_t thread_count
)
{
    // Warm-up
    if (!executor.matrix_multiply_parallel(
            A, B, C, N, thread_count)) {
        return -1.0;
    }

    if (!verify_result(C, N)) {
        return -1.0;
    }

    std::vector<double> timings;
    timings.reserve(MEASUREMENTS);

    for (int i = 0; i < MEASUREMENTS; ++i) {

        std::fill(C.begin(), C.end(), 0.0f);

        auto start = Clock::now();

        bool success =
            executor.matrix_multiply_parallel(
                A,
                B,
                C,
                N,
                thread_count
            );

        auto end = Clock::now();

        if (!success || !verify_result(C, N)) {
            return -1.0;
        }

        double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        timings.push_back(elapsed_ms);
    }

    return median(timings);
}

int main()
{
    const std::size_t elements = N * N;

    std::vector<float> A(elements, 1.0f);
    std::vector<float> B(elements, 2.0f);
    std::vector<float> C(elements, 0.0f);

    CPUExecutor cpu;

    std::cout << "============================================\n";
    std::cout << "       HeteroOS CPU Scaling Benchmark\n";
    std::cout << "============================================\n\n";

    std::cout << "Matrix size: "
              << N << " x " << N << '\n';

    std::cout << "Measurements per configuration: "
              << MEASUREMENTS << '\n';

    std::cout << "Statistic: median\n\n";

    std::cout << std::left
              << std::setw(12) << "Threads"
              << std::setw(16) << "Time (ms)"
              << std::setw(14) << "Speedup"
              << std::setw(14) << "Efficiency"
              << '\n';

    std::cout << std::string(56, '-') << '\n';

    double baseline = 0.0;

    const std::size_t thread_counts[] = {
        1, 2, 4, 6, 8, 10, 12
    };

    for (std::size_t threads : thread_counts) {

        double time_ms =
            benchmark_cpu(
                cpu,
                A,
                B,
                C,
                N,
                threads
            );

        if (time_ms < 0.0) {
            std::cerr
                << "Benchmark failed for "
                << threads
                << " threads.\n";

            return 1;
        }

        if (threads == 1) {
            baseline = time_ms;
        }

        double speedup =
            baseline / time_ms;

        double efficiency =
            speedup /
            static_cast<double>(threads);

        std::cout << std::left
                  << std::setw(12)
                  << threads

                  << std::setw(16)
                  << std::fixed
                  << std::setprecision(3)
                  << time_ms

                  << std::setw(14)
                  << std::setprecision(2)
                  << speedup
                  << "x"

                  << std::setw(14)
                  << std::setprecision(2)
                  << efficiency * 100.0
                  << "%\n";
    }

    return 0;
}