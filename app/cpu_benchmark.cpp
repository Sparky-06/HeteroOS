#include "execution/cpu_executor.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

struct BenchmarkResult {

    std::size_t threads;

    double seconds;

    double speedup;

    double efficiency;

    bool correct;
};

bool verify_result(
    const std::vector<float>& C,
    std::size_t N
) {

    const float expected =
        2.0f * static_cast<float>(N);

    for (float value : C) {

        if (std::fabs(value - expected) > 1e-3f) {
            return false;
        }
    }

    return true;
}

int main() {

    constexpr std::size_t N = 1024;

    std::cout << "========================================\n";
    std::cout << "      HeteroOS CPU Benchmark\n";
    std::cout << "========================================\n\n";

    std::cout << "Matrix size: "
              << N << " x " << N << "\n";

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C;

    for (std::size_t i = 0; i < N * N; ++i) {

        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    CPUExecutor executor;

    std::vector<BenchmarkResult> results;

    // ----------------------------------
    // SINGLE-THREAD BASELINE
    // ----------------------------------

    auto start =
        std::chrono::steady_clock::now();

    bool success =
        executor.matrix_multiply(
            A,
            B,
            C,
            N
        );

    auto end =
        std::chrono::steady_clock::now();

    if (!success) {

        std::cerr << "Baseline execution failed\n";

        return 1;
    }

    double baseline_time =
        std::chrono::duration<double>(
            end - start
        ).count();

    bool baseline_correct =
        verify_result(C, N);

    std::cout << "\nSingle-thread baseline\n";

    std::cout << "Time      : "
              << std::fixed
              << std::setprecision(6)
              << baseline_time
              << " s\n";

    std::cout << "Result    : "
              << (baseline_correct
                    ? "CORRECT"
                    : "INCORRECT")
              << "\n";

    if (!baseline_correct) {
        return 1;
    }

    // ----------------------------------
    // PARALLEL BENCHMARK
    // ----------------------------------

    const std::size_t thread_counts[] = {
        1, 2, 4, 6, 8, 10, 12
    };

    std::cout << "\n========================================\n";
    std::cout << "Parallel Scaling\n";
    std::cout << "========================================\n\n";

    std::cout
        << std::setw(8) << "Threads"
        << std::setw(14) << "Time (s)"
        << std::setw(14) << "Speedup"
        << std::setw(14) << "Efficiency"
        << '\n';

    std::cout << "----------------------------------------\n";

    for (std::size_t thread_count :
         thread_counts) {

        start =
            std::chrono::steady_clock::now();

        success =
            executor.matrix_multiply_parallel(
                A,
                B,
                C,
                N,
                thread_count
            );

        end =
            std::chrono::steady_clock::now();

        if (!success) {

            std::cerr
                << "Failed with "
                << thread_count
                << " threads\n";

            return 1;
        }

        double elapsed =
            std::chrono::duration<double>(
                end - start
            ).count();

        bool correct =
            verify_result(C, N);

        double speedup =
            baseline_time / elapsed;

        double efficiency =
            speedup /
            static_cast<double>(thread_count);

        std::cout
            << std::setw(8)
            << thread_count
            << std::setw(14)
            << std::fixed
            << std::setprecision(6)
            << elapsed
            << std::setw(14)
            << std::setprecision(2)
            << speedup
            << "x"
            << std::setw(13)
            << efficiency * 100.0
            << "%";

        if (!correct) {
            std::cout << "  INCORRECT";
        }

        std::cout << '\n';
    }

    return 0;
}