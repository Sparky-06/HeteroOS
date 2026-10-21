#ifndef CPU_EXECUTOR_HPP
#define CPU_EXECUTOR_HPP

#include <cstddef>
#include <vector>

class CPUExecutor {

private:

    void multiply_rows(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        std::size_t N,
        std::size_t row_begin,
        std::size_t row_end
    ) const;

public:

    bool matrix_multiply(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        std::size_t N
    ) const;

    bool matrix_multiply_parallel(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        std::size_t N,
        std::size_t thread_count
    ) const;

    bool vector_add(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        std::size_t N
    ) const;

    bool vector_add_parallel(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        std::size_t N,
        std::size_t thread_count
    ) const;
};

#endif