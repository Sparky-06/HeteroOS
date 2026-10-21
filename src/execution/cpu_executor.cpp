#include "execution/cpu_executor.hpp"

#include <thread>
#include <exception>
#include <algorithm>
#include <functional>

void CPUExecutor::multiply_rows(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::size_t N,
    std::size_t row_begin,
    std::size_t row_end
) const {

    for (std::size_t i = row_begin; i < row_end; ++i) {

        for (std::size_t j = 0; j < N; ++j) {

            float sum = 0.0f;

            for (std::size_t k = 0; k < N; ++k) {

                sum +=
                    A[i * N + k] *
                    B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

bool CPUExecutor::matrix_multiply(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::size_t N
) const {

    if (N == 0) {
        return false;
    }

    const std::size_t expected_size = N * N;

    if (A.size() != expected_size ||
        B.size() != expected_size) {

        return false;
    }

    C.assign(expected_size, 0.0f);

    multiply_rows(
        A,
        B,
        C,
        N,
        0,
        N
    );

    return true;
}

bool CPUExecutor::matrix_multiply_parallel(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::size_t N,
    std::size_t thread_count
) const {

    if (N == 0 || thread_count == 0) {
        return false;
    }

    const std::size_t expected_size = N * N;

    if (A.size() != expected_size ||
        B.size() != expected_size) {

        return false;
    }

    // There is no point creating more workers than rows.
    thread_count =
        std::min(thread_count, N);

    C.assign(expected_size, 0.0f);

    std::vector<std::thread> workers;

    workers.reserve(thread_count);

    // Divide rows approximately evenly.
    const std::size_t rows_per_thread =
        N / thread_count;

    const std::size_t remainder =
        N % thread_count;

    std::size_t row_begin = 0;

    for (std::size_t i = 0;
         i < thread_count;
         ++i) {

        const std::size_t extra =
            (i < remainder) ? 1 : 0;

        const std::size_t row_count =
            rows_per_thread + extra;

        const std::size_t row_end =
            row_begin + row_count;

        workers.emplace_back(
            &CPUExecutor::multiply_rows,
            this,
            std::cref(A),
            std::cref(B),
            std::ref(C),
            N,
            row_begin,
            row_end
        );

        row_begin = row_end;
    }

    for (auto& worker : workers) {
        worker.join();
    }

    return true;
}


bool CPUExecutor::vector_add(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::size_t N
) const {
    if (N == 0 ||
        A.size() != N ||
        B.size() != N ||
        C.size() != N) {
        return false;
    }

    for (std::size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }

    return true;
}


bool CPUExecutor::vector_add_parallel(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::size_t N,
    std::size_t thread_count
) const {
    if (N == 0 ||
        A.size() != N ||
        B.size() != N ||
        C.size() != N ||
        thread_count == 0) {
        return false;
    }

    // Never create more workers than there are elements.
    thread_count = std::min(thread_count, N);

    auto worker = [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            C[i] = A[i] + B[i];
        }
    };

    if (thread_count == 1) {
        worker(0, N);
        return true;
    }

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    const std::size_t base_chunk = N / thread_count;
    const std::size_t remainder = N % thread_count;

    std::size_t begin = 0;

    try {
        for (std::size_t t = 0; t < thread_count; ++t) {
            const std::size_t chunk =
                base_chunk + (t < remainder ? 1 : 0);

            const std::size_t end = begin + chunk;

            threads.emplace_back(worker, begin, end);

            begin = end;
        }
    } catch (const std::exception&) {
        // Join every thread that was successfully created.
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        return false;
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return true;
}