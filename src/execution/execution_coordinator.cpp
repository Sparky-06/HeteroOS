
#include "execution/execution_coordinator.hpp"
#include "execution/gpu_executor.hpp"
#include "kernel/kernel_event_client.hpp"

#include <chrono>
#include <exception>
#include <iostream>

bool ExecutionCoordinator::initialize_gpu()
{
    if (gpu_initialized_) {
        return true;
    }

    gpu_initialized_ = gpu_executor_.initialize();

    if (!gpu_initialized_) {
        std::cerr
            << "GPU unavailable; CPU execution remains available.\n";
    }

    return gpu_initialized_;
}

bool ExecutionCoordinator::gpu_available() const
{
    return gpu_initialized_;
}

bool ExecutionCoordinator::execute(Task& task)
{
    // Reset measurements so failed or CPU tasks cannot expose stale
    // GPU timings from a previous execution.
    last_execution_ms_ = 0.0;
    last_gpu_result_ = {};

    // The coordinator accepts only queued tasks.
    if (task.state() != TaskState::Queued) {
        std::cerr << "Task is not queued.\n";
        return false;
    }

    // A target must be selected before execution.
    if (task.target() == ExecutionTarget::Unassigned) {
        task.fail();
        std::cerr << "Task has no execution target.\n";
        return false;
    }

    // Do not start a GPU task if initialization failed.
    if (task.target() == ExecutionTarget::GPU &&
        !gpu_initialized_) {
        task.fail();
        std::cerr
            << "GPU target selected, but GPU is unavailable.\n";
        return false;
    }

    if (!task.start()) {
        std::cerr << "Failed to transition task to Running.\n";
        return false;
    }

    bool success = false;

    // Measure the executor dispatch, not output validation or scheduling.
    const auto execution_start = std::chrono::steady_clock::now();

    try {
        switch (task.type()) {
            case TaskType::VectorAdd: {
                if (task.target() == ExecutionTarget::CPU) {
                    success = cpu_executor_.vector_add(
                        task.input_a(),
                        task.input_b(),
                        task.output(),
                        task.dimension()
                    );
                } else {
                    const GPUExecutionResult result =
                        gpu_executor_.vector_add(
                            task.input_a(),
                            task.input_b(),
                            task.output(),
                            task.dimension()
                        );

                    last_gpu_result_ = result;
                    success = result.success;
                }

                break;
            }

            case TaskType::MatrixMultiply: {
                if (task.target() == ExecutionTarget::CPU) {
                    success = cpu_executor_.matrix_multiply(
                        task.input_a(),
                        task.input_b(),
                        task.output(),
                        task.dimension()
                    );
                } else {
                    const GPUExecutionResult result =
                        gpu_executor_.matrix_multiply(
                            task.input_a(),
                            task.input_b(),
                            task.output(),
                            task.dimension()
                        );

                    last_gpu_result_ = result;
                    success = result.success;
                }

                break;
            }

            default: {
                std::cerr << "Unsupported task type.\n";
                success = false;
                break;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Task execution exception: "
                  << error.what() << '\n';

        success = false;
    }

    const auto execution_end = std::chrono::steady_clock::now();

    last_execution_ms_ =
        std::chrono::duration<double, std::milli>(
            execution_end - execution_start
        ).count();

    if (!success) {
        task.fail();
        std::cerr << "Task " << task.id() << " failed.\n";
        return false;
    }

    // Execution succeeded. Output correctness is validated separately
    // by the caller, for example through verify_output(task).

    if (!task.complete()) {
        task.fail();

        std::cerr
            << "Failed to mark task as Completed.\n";

        return false;
    }

    // Best-effort kernel telemetry. A missing monitor must not affect
    // successful execution or the task's Completed state.
    if (!heteroos::record_kernel_event(task.target())) {
        std::cerr
            << "HeteroOS kernel monitor unavailable; "
            << "event was not recorded.\n";
    }

    return true;
}


double ExecutionCoordinator::last_execution_ms() const
{
    return last_execution_ms_;
}

GPUExecutionResult ExecutionCoordinator::last_gpu_result() const
{
    return last_gpu_result_;
}
