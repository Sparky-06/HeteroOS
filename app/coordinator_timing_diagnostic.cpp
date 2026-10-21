
#include "execution/execution_coordinator.hpp"
#include "task/task_manager.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

int main()
{
    constexpr std::size_t N = 256;
    constexpr int WARMUP_RUNS = 3;
    constexpr int MEASURED_RUNS = 20;
    constexpr std::size_t ELEMENTS = N * N;

    std::vector<float> A(ELEMENTS, 1.0f);
    std::vector<float> B(ELEMENTS, 2.0f);

    ExecutionCoordinator coordinator;
    TaskManager manager;

    if (!coordinator.initialize_gpu()) {
        std::cerr << "GPU initialization failed.\n";
        return 1;
    }

    std::cout << "Coordinator GPU Timing Diagnostic\n"
              << "Matrix dimension: " << N << '\n'
              << "Warm-ups: " << WARMUP_RUNS << '\n'
              << "Measured runs: " << MEASURED_RUNS << "\n\n";

    auto run_once = [&](int run, bool warmup) -> bool {
        auto task = std::make_unique<Task>(
            TaskType::MatrixMultiply, N, A, B);

        const auto id = manager.submit(std::move(task));

        if (!id) {
            std::cerr << "Task submission failed.\n";
            return false;
        }

        Task* queued_task = manager.take_next_task();

        if (queued_task == nullptr) {
            std::cerr << "Could not retrieve queued task.\n";
            return false;
        }

        if (!queued_task->set_execution_target(ExecutionTarget::GPU)) {
            std::cerr << "Could not assign GPU target. State: "
                      << to_string(queued_task->state()) << '\n';
            return false;
        }

        if (!coordinator.execute(*queued_task)) {
            std::cerr << "Execution failed at run " << run << '\n';
            return false;
        }

        const GPUExecutionResult result =
            coordinator.last_gpu_result();

        if (!result.success) {
            std::cerr << "GPU execution reported failure.\n";
            return false;
        }

        // Verify output outside the measured executor interval.
        for (float value : queued_task->output()) {
            if (std::fabs(value - static_cast<float>(2 * N)) > 1e-3f) {
                std::cerr << "Incorrect output at run " << run << '\n';
                return false;
            }
        }

        if (!warmup) {
            std::cout << std::setw(3) << run
                      << std::setw(16)
                      << coordinator.last_execution_ms()
                      << std::setw(16) << result.total_ms
                      << std::setw(16) << result.kernel_ms
                      << '\n';
        }

        return true;
    };

    std::cout << "Running warm-ups...\n";

    for (int i = 1; i <= WARMUP_RUNS; ++i) {
        if (!run_once(i, true)) {
            return 1;
        }
    }

    std::cout << '\n'
              << std::setw(3) << "Run"
              << std::setw(16) << "Executor(ms)"
              << std::setw(16) << "GPU total(ms)"
              << std::setw(16) << "Kernel(ms)"
              << '\n';

    std::cout << std::fixed << std::setprecision(4);

    for (int i = 1; i <= MEASURED_RUNS; ++i) {
        if (!run_once(i, false)) {
            return 1;
        }
    }

    return 0;
}
