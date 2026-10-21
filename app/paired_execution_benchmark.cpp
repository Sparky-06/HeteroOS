#include "execution/cpu_executor.hpp"
#include "execution/gpu_executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

using Clock = std::chrono::steady_clock;

constexpr int WARMUP_RUNS = 3;
constexpr int MEASURED_RUNS = 20;

struct TimingSummary {
    double median_ms = 0.0;
    double p95_ms = 0.0;
};

struct WorkloadResult {
    std::string type;
    std::size_t dimension = 0;
    TimingSummary cpu;
    TimingSummary gpu;
    double speedup = 0.0;
    std::string measured_winner;
    bool correct = false;
};

template <typename Function>
double measure_ms(Function&& function)
{
    const auto start = Clock::now();

    function();

    const auto end = Clock::now();

    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

TimingSummary summarize(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());

    const std::size_t n = samples.size();

    // Median: average the middle two values for even sample counts.
    double median = 0.0;

    if (n % 2 == 0) {
        median = (samples[n / 2 - 1] + samples[n / 2]) / 2.0;
    } else {
        median = samples[n / 2];
    }

    // Nearest-rank P95.
    const std::size_t p95_index =
        static_cast<std::size_t>(
            std::ceil(0.95 * static_cast<double>(n))
        ) - 1;

    return {median, samples[p95_index]};
}

bool verify_vector(
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C)
{
    if (A.size() != B.size() || A.size() != C.size()) {
        return false;
    }

    for (std::size_t i = 0; i < A.size(); ++i) {
        if (std::fabs(C[i] - (A[i] + B[i])) > 1e-5f) {
            return false;
        }
    }

    return true;
}

bool verify_matrix(
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C,
    std::size_t N)
{
    if (A.size() != N * N ||
        B.size() != N * N ||
        C.size() != N * N) {
        return false;
    }

    for (std::size_t row = 0; row < N; ++row) {
        for (std::size_t col = 0; col < N; ++col) {
            float expected = 0.0f;

            for (std::size_t k = 0; k < N; ++k) {
                expected += A[row * N + k] * B[k * N + col];
            }

            if (std::fabs(
                    C[row * N + col] - expected) > 1e-3f) {
                return false;
            }
        }
    }

    return true;
}

WorkloadResult benchmark_vector(
    CPUExecutor& cpu,
    GPUExecutor& gpu,
    std::size_t N)
{
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> cpu_output(N);
    std::vector<float> gpu_output(N);

    auto run_cpu = [&]() {
        if (!cpu.vector_add(A, B, cpu_output, N)) {
            throw std::runtime_error("CPU vector addition failed");
        }
    };

    auto run_gpu = [&]() {
        const GPUExecutionResult result =
            gpu.vector_add(A, B, gpu_output, N);

        if (!result.success) {
            throw std::runtime_error("GPU vector addition failed");
        }
    };

    // Warm up both implementations before collecting samples.
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        run_cpu();
        run_gpu();
    }

    std::vector<double> cpu_samples;
    std::vector<double> gpu_samples;

    cpu_samples.reserve(MEASURED_RUNS);
    gpu_samples.reserve(MEASURED_RUNS);

    // Alternate order to reduce systematic order bias.
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        if (i % 2 == 0) {
            cpu_samples.push_back(measure_ms(run_cpu));
            gpu_samples.push_back(measure_ms(run_gpu));
        } else {
            gpu_samples.push_back(measure_ms(run_gpu));
            cpu_samples.push_back(measure_ms(run_cpu));
        }
    }

    const bool correct =
        verify_vector(A, B, cpu_output) &&
        verify_vector(A, B, gpu_output);

    const TimingSummary cpu_timing = summarize(cpu_samples);
    const TimingSummary gpu_timing = summarize(gpu_samples);

    WorkloadResult result;
    result.type = "VectorAdd";
    result.dimension = N;
    result.cpu = cpu_timing;
    result.gpu = gpu_timing;
    result.speedup = cpu_timing.median_ms / gpu_timing.median_ms;
    result.measured_winner =
        gpu_timing.median_ms < cpu_timing.median_ms ? "GPU" : "CPU";
    result.correct = correct;

    return result;
}

WorkloadResult benchmark_matrix(
    CPUExecutor& cpu,
    GPUExecutor& gpu,
    std::size_t N)
{
    const std::size_t elements = N * N;

    std::vector<float> A(elements, 1.0f);
    std::vector<float> B(elements, 2.0f);
    std::vector<float> cpu_output(elements);
    std::vector<float> gpu_output(elements);

    auto run_cpu = [&]() {
        if (!cpu.matrix_multiply(A, B, cpu_output, N)) {
            throw std::runtime_error("CPU matrix multiplication failed");
        }
    };

    auto run_gpu = [&]() {
        const GPUExecutionResult result =
            gpu.matrix_multiply(A, B, gpu_output, N);

        if (!result.success) {
            throw std::runtime_error("GPU matrix multiplication failed");
        }
    };

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        run_cpu();
        run_gpu();
    }

    std::vector<double> cpu_samples;
    std::vector<double> gpu_samples;

    cpu_samples.reserve(MEASURED_RUNS);
    gpu_samples.reserve(MEASURED_RUNS);

    for (int i = 0; i < MEASURED_RUNS; ++i) {
        if (i % 2 == 0) {
            cpu_samples.push_back(measure_ms(run_cpu));
            gpu_samples.push_back(measure_ms(run_gpu));
        } else {
            gpu_samples.push_back(measure_ms(run_gpu));
            cpu_samples.push_back(measure_ms(run_cpu));
        }
    }

    const bool correct =
        verify_matrix(A, B, cpu_output, N) &&
        verify_matrix(A, B, gpu_output, N);

    const TimingSummary cpu_timing = summarize(cpu_samples);
    const TimingSummary gpu_timing = summarize(gpu_samples);

    WorkloadResult result;
    result.type = "MatrixMultiply";
    result.dimension = N;
    result.cpu = cpu_timing;
    result.gpu = gpu_timing;
    result.speedup = cpu_timing.median_ms / gpu_timing.median_ms;
    result.measured_winner =
        gpu_timing.median_ms < cpu_timing.median_ms ? "GPU" : "CPU";
    result.correct = correct;

    return result;
}

void print_result(const WorkloadResult& result)
{
    std::cout << std::left
              << std::setw(16) << result.type
              << std::setw(12) << result.dimension
              << std::right << std::fixed << std::setprecision(4)
              << std::setw(12) << result.cpu.median_ms
              << std::setw(12) << result.gpu.median_ms
              << std::setw(12) << result.cpu.p95_ms
              << std::setw(12) << result.gpu.p95_ms
              << std::setw(12) << result.speedup
              << "x  "
              << std::setw(6) << result.measured_winner
              << "  "
              << (result.correct ? "PASS" : "FAIL")
              << '\n';
}

void write_csv_header(std::ofstream& csv)
{
    csv << "type,dimension,samples,"
        << "cpu_median_ms,gpu_median_ms,"
        << "cpu_p95_ms,gpu_p95_ms,"
        << "cpu_over_gpu_speedup,measured_winner,correct\n";
}

void write_csv_row(
    std::ofstream& csv,
    const WorkloadResult& result,
    const std::string& condition)
{
    csv << condition << ','
        << result.type << ','
        << result.dimension << ','
        << MEASURED_RUNS << ','
        << result.cpu.median_ms << ','
        << result.gpu.median_ms << ','
        << result.cpu.p95_ms << ','
        << result.gpu.p95_ms << ','
        << result.speedup << ','
        << result.measured_winner << ','
        << (result.correct ? "true" : "false")
        << '\n';
}

int main(int argc, char* argv[])
{
    const std::string condition =
        argc >= 2 ? argv[1] : "idle";

    if (condition != "idle" && condition != "ollama") {
        std::cerr << "Usage: " << argv[0]
                  << " [idle|ollama]\n";
        return 2;
    }

    try {
        CPUExecutor cpu;
        GPUExecutor gpu;

        if (!gpu.initialize()) {
            std::cerr << "GPU initialization failed\n";
            return 1;
        }

        std::ofstream csv(
            "build/paired_execution_benchmark.csv",
            std::ios::trunc
        );

        if (!csv) {
            std::cerr << "Could not open CSV report\n";
            return 1;
        }

        csv << "condition,type,dimension,samples,"
            << "cpu_median_ms,gpu_median_ms,"
            << "cpu_p95_ms,gpu_p95_ms,"
            << "cpu_over_gpu_speedup,measured_winner,correct\n";

        std::cout << "HeteroOS Paired CPU/GPU Benchmark\n"
                  << "Warm-ups per executor: " << WARMUP_RUNS << '\n'
                  << "Measured samples: " << MEASURED_RUNS << '\n'
                  << "Statistic: median and nearest-rank P95\n\n";

        std::cout << std::left
                  << std::setw(16) << "Workload"
                  << std::setw(12) << "Dimension"
                  << std::right
                  << std::setw(12) << "CPU median"
                  << std::setw(12) << "GPU median"
                  << std::setw(12) << "CPU P95"
                  << std::setw(12) << "GPU P95"
                  << std::setw(12) << "Speedup"
                  << "  Winner  Correct\n";

        std::cout << std::string(104, '-') << '\n';

        const std::vector<WorkloadResult> results{
            benchmark_vector(cpu, gpu, 65536),
            benchmark_vector(cpu, gpu, 1048576),
            benchmark_matrix(cpu, gpu, 256),
            benchmark_matrix(cpu, gpu, 512)
        };

        bool all_correct = true;

        for (const auto& result : results) {
            print_result(result);
            write_csv_row(csv, result, condition);
            all_correct = all_correct && result.correct;
        }

        csv.flush();

        if (!csv) {
            std::cerr << "Failed to write CSV report\n";
            return 1;
        }

        std::cout << "\nCSV: build/paired_execution_benchmark.csv\n";

        gpu.shutdown();

        if (!all_correct) {
            std::cerr << "One or more correctness checks failed\n";
            return 1;
        }

        std::cout << "All correctness checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}