
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

constexpr int WARMUP_RUNS = 3;
constexpr int MEASURED_RUNS = 20;

struct TimingSummary {
    double median_ms;
    double p95_ms;
};

TimingSummary summarize(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());

    const std::size_t n = samples.size();

    double median_ms;
    if (n % 2 == 0) {
        median_ms =
            (samples[n / 2 - 1] + samples[n / 2]) / 2.0;
    } else {
        median_ms = samples[n / 2];
    }

    const std::size_t p95_index =
        static_cast<std::size_t>(
            std::ceil(0.95 * static_cast<double>(n))
        ) - 1;

    return {median_ms, samples[p95_index]};
}

bool verify_copy(
    const std::vector<unsigned char>& source,
    const std::vector<unsigned char>& destination)
{
    return source.size() == destination.size()
        && std::memcmp(
            source.data(),
            destination.data(),
            source.size()
        ) == 0;
}

struct BenchmarkResult {
    std::size_t bytes;
    TimingSummary timing;
    double bandwidth_gbps;
    bool correct;
};

BenchmarkResult benchmark_size(std::size_t bytes)
{
    std::vector<unsigned char> source(bytes);
    std::vector<unsigned char> destination(bytes);

    for (std::size_t i = 0; i < bytes; ++i) {
        source[i] =
            static_cast<unsigned char>((i * 31 + 7) & 0xff);
    }

    std::fill(destination.begin(), destination.end(), 0);

    auto run_copy = [&]() {
        std::memcpy(
            destination.data(),
            source.data(),
            bytes
        );
    };

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        run_copy();
    }

    std::vector<double> samples;
    samples.reserve(MEASURED_RUNS);

    for (int i = 0; i < MEASURED_RUNS; ++i) {
        const auto start = Clock::now();
        run_copy();
        const auto end = Clock::now();

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        if (elapsed_ms <= 0.0) {
            throw std::runtime_error(
                "Invalid timing measurement"
            );
        }

        samples.push_back(elapsed_ms);
    }

    const bool correct = verify_copy(source, destination);
    const TimingSummary timing = summarize(samples);

    // Count one read and one write for each copied byte.
    const double bytes_moved = 2.0 * static_cast<double>(bytes);
    const double elapsed_seconds = timing.median_ms / 1000.0;

    const double bandwidth_gbps =
        bytes_moved / elapsed_seconds / 1e9;

    return {bytes, timing, bandwidth_gbps, correct};
}

int main(int argc, char* argv[])
{
    const std::string csv_path =
        argc > 1
            ? argv[1]
            : "build/memory_bandwidth_benchmark.csv";

    // Each size is the size of ONE array, not both arrays combined.
    const std::size_t sizes[] = {
        256ULL * 1024,
        1ULL * 1024 * 1024,
        8ULL * 1024 * 1024,
        32ULL * 1024 * 1024,
        128ULL * 1024 * 1024
    };

    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "Cannot open CSV: " << csv_path << '\n';
        return 1;
    }

    csv << "buffer_bytes,warmup_runs,measured_runs,"
        << "median_ms,p95_ms,effective_copy_gbps,correct\n";

    std::cout << "============================================\n"
              << " HeteroOS Memory Bandwidth Benchmark\n"
              << "============================================\n"
              << "Warm-ups: " << WARMUP_RUNS << '\n'
              << "Measured runs: " << MEASURED_RUNS << '\n'
              << "Bandwidth counts reads + writes.\n\n";

    std::cout << std::left
              << std::setw(14) << "Buffer (MiB)"
              << std::setw(14) << "Median (ms)"
              << std::setw(14) << "P95 (ms)"
              << std::setw(18) << "Copy GB/s"
              << "Result\n";

    try {
        for (const std::size_t bytes : sizes) {
            const BenchmarkResult result = benchmark_size(bytes);

            std::cout << std::fixed << std::setprecision(3)
                      << std::left
                      << std::setw(14)
                      << (static_cast<double>(bytes) / (1024.0 * 1024.0))
                      << std::setw(14) << result.timing.median_ms
                      << std::setw(14) << result.timing.p95_ms
                      << std::setw(18) << result.bandwidth_gbps
                      << (result.correct ? "PASS" : "FAIL")
                      << '\n';

            csv << result.bytes << ','
                << WARMUP_RUNS << ','
                << MEASURED_RUNS << ','
                << std::setprecision(8)
                << result.timing.median_ms << ','
                << result.timing.p95_ms << ','
                << result.bandwidth_gbps << ','
                << (result.correct ? "true" : "false")
                << '\n';

            if (!result.correct) {
                std::cerr << "Copy verification failed for "
                          << bytes << " bytes\n";
                return 1;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }

    csv.close();

    if (!csv) {
        std::cerr << "Failed while writing CSV.\n";
        return 1;
    }

    std::cout << "\nAll copy checks passed.\n"
              << "CSV saved to: " << csv_path << '\n';

    return 0;
}
