
#include "execution/execution_coordinator.hpp"
#include "profiling/profiler.hpp"
#include "task/task_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

bool verify_output(const Task& task)
{
    const auto& A = task.input_a();
    const auto& B = task.input_b();
    const auto& C = task.output();
    const std::size_t N = task.dimension();

    if (task.type() == TaskType::VectorAdd) {
        for (std::size_t i = 0; i < N; ++i) {
            if (std::fabs(C[i] - (A[i] + B[i])) > 1e-5f) {
                return false;
            }
        }
        return true;
    }

    if (task.type() == TaskType::MatrixMultiply) {
        for (std::size_t row = 0; row < N; ++row) {
            for (std::size_t col = 0; col < N; ++col) {
                float expected = 0.0f;

                for (std::size_t k = 0; k < N; ++k) {
                    expected += A[row * N + k] * B[k * N + col];
                }

                if (std::fabs(C[row * N + col] - expected) > 1e-3f) {
                    return false;
                }
            }
        }
        return true;
    }

    return false;
}

std::unique_ptr<Task> make_task(TaskType type, std::size_t N)
{
    const std::size_t elements =
        type == TaskType::VectorAdd ? N : N * N;

    return std::make_unique<Task>(
        type,
        N,
        std::vector<float>(elements, 1.0f),
        std::vector<float>(elements, 2.0f)
    );
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;

    if (values.size() % 2 != 0) {
        return values[middle];
    }

    return (values[middle - 1] + values[middle]) / 2.0;
}

void print_profile(
    const Profiler& profiler,
    TaskType type,
    std::size_t N,
    ExecutionTarget target)
{
    const auto count = profiler.sample_count(type, N, target);
    const auto estimate = profiler.estimate_ms(type, N, target);

    std::cout << "  " << to_string(type)
              << " N=" << N
              << " | " << to_string(target)
              << " | samples=" << count;

    if (estimate) {
        std::cout << " | median=" << *estimate << " ms";
    } else {
        std::cout << " | insufficient samples";
    }

    std::cout << '\n';
}

int main()
{
    constexpr int WARMUP_RUNS = 1;
    constexpr int MEASURED_RUNS = 5;

    TaskManager manager;
    ExecutionCoordinator coordinator;
    Profiler profiler;

    const bool gpu_available = coordinator.initialize_gpu();

    std::cout << "HeteroOS profiler calibration\n"
              << "GPU available: "
              << (gpu_available ? "yes" : "no") << "\n\n";

    const std::vector<std::pair<TaskType, std::size_t>> workloads = {
        {TaskType::VectorAdd, 65536},
        {TaskType::MatrixMultiply, 256}
    };

    std::cout << std::fixed << std::setprecision(4);

    for (const auto& [type, N] : workloads) {
        std::cout << "Workload: " << to_string(type)
                  << " | N=" << N << '\n';

        std::vector<ExecutionTarget> targets = {
            ExecutionTarget::CPU
        };

        if (gpu_available) {
            targets.push_back(ExecutionTarget::GPU);
        }

        for (const auto target : targets) {
            std::vector<double> measured_times;
            bool target_failed = false;

            for (int run = 0;
                 run < WARMUP_RUNS + MEASURED_RUNS;
                 ++run) {
                auto task = make_task(type, N);
                const auto id = manager.submit(std::move(task));

                if (!id) {
                    std::cerr << "Task submission failed\n";
                    return 1;
                }

                Task* queued_task = manager.take_next_task();

                if (queued_task == nullptr ||
                    !queued_task->set_execution_target(target)) {
                    std::cerr << "Could not assign execution target\n";
                    return 1;
                }

                if (!coordinator.execute(*queued_task)) {
                    std::cerr << "Execution failed: "
                              << to_string(type) << " on "
                              << to_string(target) << '\n';
                    target_failed = true;
                    break;
                }

                if (!verify_output(*queued_task)) {
                    std::cerr << "Output validation failed: "
                              << to_string(type) << " on "
                              << to_string(target) << '\n';
                    target_failed = true;
                    break;
                }

                const double execution_ms =
                    coordinator.last_execution_ms();

                if (run >= WARMUP_RUNS) {
                    measured_times.push_back(execution_ms);

                    profiler.record({
                        type,
                        N,
                        target,
                        execution_ms,
                        true
                    });
                }
            }

            if (target_failed) {
                std::cout << "  " << to_string(target)
                          << ": calibration stopped after failure\n";
                continue;
            }

            if (!measured_times.empty()) {
                const auto [minimum, maximum] =
                    std::minmax_element(
                        measured_times.begin(),
                        measured_times.end()
                    );

                std::cout << "  " << to_string(target)
                          << " | median=" << median(measured_times)
                          << " ms"
                          << " | min=" << *minimum
                          << " ms"
                          << " | max=" << *maximum
                          << " ms"
                          << " | samples=" << measured_times.size()
                          << '\n';
            }
        }

        std::cout << '\n';
    }

    std::cout << "Historical profile estimates:\n";

    for (const auto& [type, N] : workloads) {
        print_profile(profiler, type, N, ExecutionTarget::CPU);

        if (gpu_available) {
            print_profile(profiler, type, N, ExecutionTarget::GPU);
        }
    }

    std::cout << "\nCalibration complete.\n";

    if (!profiler.save_csv("build/profiles.csv")) {
        std::cerr << "Failed to save profiling data\n";
        return 1;
    }
 
    std::cout << "Saved profiling data to build/profiles.csv\n";

    return 0;
}
