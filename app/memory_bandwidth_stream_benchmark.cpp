
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
constexpr std::size_t BUFFER_SIZE = 8ULL * 1024 * 1024;

struct TimingSummary {
    double median_ms;
    double p95_ms;
};

struct BenchmarkResult {
    std::size_t pair_count;
    std::size_t total_bytes;
    TimingSummary timing;
    double effective_copy_gbps;
    bool correct;
};

struct BufferPair {
    std::vector<unsigned char> source;
    std::vector<unsigned char> destination;

    explicit BufferPair(std::size_t bytes)
        : source(bytes), destination(bytes, 0)
    {
        for (std::size_t i = 0; i < bytes; ++i) {
            source[i] =
                static_cast<unsigned char>((i * 31 + 7) & 0xff);
        }
    }
};

TimingSummary summarize(std::vector<double> samples)
{
    if (samples.empty()) {
        throw std::runtime_error("No timing samples collected");
    }

    std::sort(samples.begin(), samples.end());

    const std::size_t n = samples.size();

    const double median_ms = n % 2 == 0
        ? (samples[n / 2 - 1] + samples[n / 2]) / 2.0
        : samples[n / 2];

    const std::size_t p95_index =
        static_cast<std::size_t>(
            std::ceil(0.95 * static_cast<double>(n))
        ) - 1;

    return {median_ms, samples[p95_index]};
}

bool verify_pair(const BufferPair& pair)
{
    return std::memcmp(
        pair.source.data(),
        pair.destination.data(),
        pair.source.size()
    ) == 0;
}

BenchmarkResult benchmark_configuration(std::size_t pair_count)
{
    std::vector<BufferPair> pairs;
    pairs.reserve(pair_count);

    for (std::size_t i = 0; i < pair_count; ++i) {
        pairs.emplace_back(BUFFER_SIZE);
    }

    // Copy every pair once per measured iteration.
    // The timing covers the complete batch of copies.
    auto run_batch = [&]() {
        for (BufferPair& pair : pairs) {
            std::memcpy(
                pair.destination.data(),
                pair.source.data(),
                BUFFER_SIZE
            );
        }
    };

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        run_batch();
    }

    std::vector<double> samples;
    samples.reserve(MEASURED_RUNS);

    for (int i = 0; i < MEASURED_RUNS; ++i) {
        const auto start = Clock::now();
        run_batch();
        const auto end = Clock::now();

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        if (elapsed_ms <= 0.0) {
            throw std::runtime_error("Invalid timing measurement");
        }

        samples.push_back(elapsed_ms);
    }

    bool correct = true;
    for (const BufferPair& pair : pairs) {
        correct = correct && verify_pair(pair);
    }

    const TimingSummary timing = summarize(samples);

    // Each pair reads BUFFER_SIZE bytes and writes BUFFER_SIZE bytes.
    const std::size_t total_bytes =
        pair_count * BUFFER_SIZE * 2;

    const double effective_gbps =
        static_cast<double>(total_bytes)
        / (timing.median_ms / 1000.0)
        / 1e9;

    return {
        pair_count,
        total_bytes,
        timing,
        effective_gbps,
        correct
    };
}

int main(int argc, char* argv[])
{
    const std::string csv_path = argc > 1
        ? argv[1]
        : "build/memory_bandwidth_stream_benchmark.csv";

    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "Cannot open CSV: " << csv_path << '\n';
        return 1;
    }

    csv << "buffer_pairs,bytes_per_array,total_bytes,"
        << "warmup_runs,measured_runs,median_ms,p95_ms,"
        << "effective_copy_gbps,correct\n";

    const std::size_t pair_counts[] = {1, 2, 4, 8};

    std::cout
        << "============================================\n"
        << " HeteroOS Multi-Buffer Copy Benchmark\n"
        << "============================================\n"
        << "Buffer size per array: 8 MiB\n"
        << "Warm-ups: " << WARMUP_RUNS << '\n'
        << "Measured runs: " << MEASURED_RUNS << "\n\n";

    std::cout << std::left
              << std::setw(12) << "Pairs"
              << std::setw(14) << "Working MiB"
              << std::setw(14) << "Median ms"
              << std::setw(14) << "P95 ms"
              << std::setw(18) << "Copy GB/s"
              << "Result\n";

    try {
        for (const std::size_t count : pair_counts) {
            const BenchmarkResult result =
                benchmark_configuration(count);

            const double working_mib =
                static_cast<double>(result.total_bytes)
                / (1024.0 * 1024.0);

            std::cout << std::fixed << std::setprecision(3)
                      << std::left
                      << std::setw(12) << result.pair_count
                      << std::setw(14) << working_mib
                      << std::setw(14) << result.timing.median_ms
                      << std::setw(14) << result.timing.p95_ms
                      << std::setw(18) << result.effective_copy_gbps
                      << (result.correct ? "PASS" : "FAIL")
                      << '\n';

            csv << result.pair_count << ','
                << BUFFER_SIZE << ','
                << result.total_bytes << ','
                << WARMUP_RUNS << ','
                << MEASURED_RUNS << ','
                << std::setprecision(8)
                << result.timing.median_ms << ','
                << result.timing.p95_ms << ','
                << result.effective_copy_gbps << ','
                << (result.correct ? "true" : "false")
                << '\n';

            if (!result.correct) {
                std::cerr
                    << "Verification failed for "
                    << count << " buffer pairs\n";
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
        std::cerr << "Failed while writing CSV\n";
        return 1;
    }

    std::cout << "\nAll copy checks passed.\n"
              << "CSV saved to: " << csv_path << '\n';

    return 0;
}
