#ifndef HETEROOS_EXECUTION_COORDINATOR_HPP
#define HETEROOS_EXECUTION_COORDINATOR_HPP

#include "execution/cpu_executor.hpp"
#include "execution/gpu_executor.hpp"
#include "task/task.hpp"
#include <chrono>

class ExecutionCoordinator {
public:
    ExecutionCoordinator() = default;
    ~ExecutionCoordinator() = default;

    double last_execution_ms() const;

    GPUExecutionResult last_gpu_result() const;

    ExecutionCoordinator(const ExecutionCoordinator&) = delete;
    ExecutionCoordinator& operator=(const ExecutionCoordinator&) = delete;

    // GPU availability is optional: CPU execution can still work.
    bool initialize_gpu();

    bool gpu_available() const;

    // Executes a queued task on its assigned target.
    bool execute(Task& task);

private:
    CPUExecutor cpu_executor_;
    GPUExecutor gpu_executor_;

    double last_execution_ms_ = 0.0;
    GPUExecutionResult last_gpu_result_{};

    bool gpu_initialized_ = false;
};

#endif