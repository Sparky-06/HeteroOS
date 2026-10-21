#include "execution/execution_coordinator.hpp"
#include "scheduling/scheduler.hpp"
#include "task/task_manager.hpp"
#include "monitoring/resource_monitor.hpp"

#include "profiling/profiler.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <utility>

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

                if (std::fabs(
                        C[row * N + col] - expected) > 1e-3f) {
                    return false;
                }
            }
        }

        return true;
    }

    return false;
}

void print_estimate(
    const char* label,
    const std::optional<double>& estimate
)
{
    std::cout << label;

    if (estimate) {
        std::cout << std::fixed << std::setprecision(4)
                  << *estimate << " ms";
    } else {
        std::cout << "unavailable";
    }

    std::cout << '\n';
}


void write_optional(
    std::ostream& output,
    const std::optional<double>& value
) {
    if (value.has_value()) {
        output << *value;
    }
}



int main()
{
    using Clock = std::chrono::steady_clock;

    

    std::ofstream report(
        "build/scheduling_measurements.csv",
        std::ios::trunc
    );

    if (!report) {
        std::cerr << "Could not open scheduling measurement report\n";
        return 1;
    }

    report << std::fixed << std::setprecision(6);
    report
        << "task_id,task_type,dimension,target,"
        << "base_cpu_ms,base_gpu_ms,"
        << "estimated_cpu_ms,estimated_gpu_ms,"
        << "scheduling_ms,executor_ms,validation_ms,total_task_ms,"
        << "predicted_selected_ms,absolute_error_ms,"
        << "relative_error_percent,correct\n";

    TaskManager manager;
    ExecutionCoordinator coordinator;
    Profiler profiler;

    if (profiler.load_csv("build/profiles.csv")) {
        std::cout << "Loaded historical profiling data.\n";
    } else {
        std::cout << "No valid historical profile loaded; "
                << "using benchmark fallbacks.\n";
    }


    Scheduler scheduler(profiler);
    ResourceMonitor resource_monitor;

    resource_monitor.initialize();

    const bool gpu_available = coordinator.initialize_gpu();

    // Establish the initial CPU utilization baseline.
    resource_monitor.collect();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    SystemSnapshot snapshot = resource_monitor.collect();

    std::cout << "\nGPU available: "
              << (gpu_available ? "yes" : "no") << '\n';

    auto submit_task = [&](std::unique_ptr<Task> task) -> bool {
        const auto id = manager.submit(std::move(task));

        if (!id) {
            std::cerr << "Task submission failed.\n";
            return false;
        }

        std::cout << "Submitted task " << *id << '\n';
        return true;
    };

    constexpr std::size_t VECTOR_N = 65536;

    if (!submit_task(std::make_unique<Task>(
            TaskType::VectorAdd,
            VECTOR_N,
            std::vector<float>(VECTOR_N, 1.0f),
            std::vector<float>(VECTOR_N, 2.0f)))) {
        return 1;
    }

    constexpr std::size_t MATRIX_N = 256;
    constexpr std::size_t MATRIX_ELEMENTS = MATRIX_N * MATRIX_N;

    if (!submit_task(std::make_unique<Task>(
            TaskType::MatrixMultiply,
            MATRIX_N,
            std::vector<float>(MATRIX_ELEMENTS, 1.0f),
            std::vector<float>(MATRIX_ELEMENTS, 2.0f)))) {
        return 1;
    }

    std::cout << "\nScheduling and execution:\n";

    bool all_tasks_correct = true;

    
    while (!manager.empty()) {
        Task* task = manager.take_next_task();

        if (task == nullptr) {
            break;
        }

        // Keep resource sampling outside the measured task interval.
        std::this_thread::sleep_for(std::chrono::seconds(1));
        snapshot = resource_monitor.collect();

        // Total interval: scheduling + execution + validation.
        const auto task_start = Clock::now();

        // Measure only the scheduling decision.
        const auto scheduling_start = Clock::now();

        const ScheduleDecision decision =
            scheduler.choose_target(*task, snapshot, gpu_available);

        const auto scheduling_end = Clock::now();

        const double scheduling_ms =
            std::chrono::duration<double, std::milli>(
                scheduling_end - scheduling_start
            ).count();

        std::cout << "\nTask " << task->id()
                << " | " << to_string(task->type())
                << " | dimension: " << task->dimension() << '\n';

        std::cout << "  CPU telemetry: ";

        if (snapshot.cpu_valid) {
            std::cout << snapshot.cpu.total_usage << "% usage\n";
        } else {
            std::cout << "unavailable\n";
        }

        std::cout << "  GPU telemetry: ";

        if (snapshot.gpu_valid) {
            std::cout << snapshot.gpu.gpu_utilization
                    << "% utilization, "
                    << snapshot.gpu.memory_free / (1024 * 1024)
                    << " MiB free VRAM\n";
        } else {
            std::cout << "unavailable\n";
        }

        std::cout << "  Benchmark costs:\n";
        print_estimate("    CPU base: ", decision.base_cpu_ms);
        print_estimate("    GPU base: ", decision.base_gpu_ms);

        std::cout << "  Resource-adjusted costs:\n";
        print_estimate("    CPU: ", decision.estimated_cpu_ms);
        print_estimate("    GPU: ", decision.estimated_gpu_ms);

        std::cout << "  GPU eligible: "
                << (decision.gpu_eligible ? "yes" : "no") << '\n';

        if (!decision.gpu_eligible) {
            std::cout << "  GPU rejection reason: "
                    << decision.gpu_ineligibility_reason << '\n';
        }

        std::cout << "  Selected target: "
                << to_string(decision.target) << '\n';

        std::cout << "  Decision reason: "
                << decision.reason << '\n';

        if (!task->set_execution_target(decision.target)) {
            std::cerr << "Failed to assign execution target.\n";
            return 1;
        }

        // Execute exactly once.
        const bool executed = coordinator.execute(*task);
        const double executor_ms = coordinator.last_execution_ms();

        if (!executed) {
            std::cerr << "Execution failed for task "
                    << task->id() << '\n';
            all_tasks_correct = false;
            return 1;
        }

        // Measure validation separately.
        const auto validation_start = Clock::now();

        const bool correct = verify_output(*task);

        const auto validation_end = Clock::now();

        const double validation_ms =
            std::chrono::duration<double, std::milli>(
                validation_end - validation_start
            ).count();

        // Includes scheduling, execution, validation, and intervening
        // orchestration overhead. Excludes resource sampling and submission.
        const double total_task_ms =
            std::chrono::duration<double, std::milli>(
                validation_end - task_start
            ).count();

        if (!correct) {
            std::cerr << "Output verification failed for task "
                    << task->id() << '\n';
            all_tasks_correct = false;
        }

        const std::optional<double>& predicted_ms =
            decision.target == ExecutionTarget::GPU
                ? decision.estimated_gpu_ms
                : decision.estimated_cpu_ms;

        std::optional<double> absolute_error_ms;
        std::optional<double> relative_error_percent;

        if (predicted_ms.has_value()) {
            absolute_error_ms =
                std::abs(executor_ms - *predicted_ms);

            if (*predicted_ms > 0.0) {
                relative_error_percent =
                    (*absolute_error_ms / *predicted_ms) * 100.0;
            }
        }

        // Write CSV row.
        report << task->id() << ','
            << to_string(task->type()) << ','
            << task->dimension() << ','
            << to_string(decision.target) << ',';

        write_optional(report, decision.base_cpu_ms);
        report << ',';
        write_optional(report, decision.base_gpu_ms);
        report << ',';
        write_optional(report, decision.estimated_cpu_ms);
        report << ',';
        write_optional(report, decision.estimated_gpu_ms);

        report << ','
            << scheduling_ms << ','
            << executor_ms << ','
            << validation_ms << ','
            << total_task_ms << ',';

        write_optional(report, predicted_ms);
        report << ',';
        write_optional(report, absolute_error_ms);
        report << ',';
        write_optional(report, relative_error_percent);

        report << ',' << (correct ? "true" : "false") << '\n';

        std::cout << "  Scheduling time: " << scheduling_ms << " ms\n"
                << "  Executor time:   " << executor_ms << " ms\n"
                << "  Validation time: " << validation_ms << " ms\n"
                << "  Total task time: " << total_task_ms << " ms\n";

        if (predicted_ms.has_value()) {
            std::cout << "  Predicted selected-target cost: "
                    << *predicted_ms << " ms\n";

            std::cout << "  Absolute prediction error: "
                    << *absolute_error_ms << " ms\n";

            if (relative_error_percent.has_value()) {
                std::cout << "  Relative prediction error: "
                        << *relative_error_percent << "%\n";
            }
        }

        std::cout << "  Final state: "
                << to_string(task->state())
                << " | Output "
                << (correct ? "verified" : "INVALID") << '\n';
        
        if (correct) {
            profiler.record({
                task->type(),
                task->dimension(),
                task->target(),
                coordinator.last_execution_ms(),
                true
            });

            if (!profiler.save_csv("build/profiles.csv")) {
                std::cerr << "Warning: failed to save updated profiling data\n";
            }
        }
    }



    std::cout << "\nHistorical profiling results:\n";

    for (const auto& workload : {
            std::pair{TaskType::VectorAdd, std::size_t{65536}},
            std::pair{TaskType::MatrixMultiply, std::size_t{256}}
        }) {

        for (const auto target : {
                ExecutionTarget::CPU,
                ExecutionTarget::GPU
            }) {

            const auto samples = profiler.sample_count(
                workload.first, workload.second, target);

            const auto estimate = profiler.estimate_ms(
                workload.first, workload.second, target);

            std::cout << "  " << to_string(workload.first)
                    << " N=" << workload.second
                    << " on " << to_string(target)
                    << ": samples=" << samples;

            if (estimate) {
                std::cout << ", median=" << *estimate << " ms";
            } else {
                std::cout << ", insufficient samples";
            }

            std::cout << '\n';
        }
    }


    std::cout << "\nFinal task states:\n";

    for (std::uint64_t id = 1; id <= manager.task_count(); ++id) {
        const Task* task = manager.get_task(id);

        if (task == nullptr) {
            continue;
        }

        std::cout << "Task " << id
                  << ": " << to_string(task->state())
                  << " on " << to_string(task->target()) << '\n';

        if (task->target() == ExecutionTarget::GPU) {
            const GPUExecutionResult result =
            coordinator.last_gpu_result();

            const double component_sum_ms =
                result.allocation_ms +
                result.host_to_device_ms +
                result.kernel_ms +
                result.device_to_host_ms +
                result.cleanup_ms;

           std::cout << "GPU timing breakdown:\n"
                << "  Allocation:     " << result.allocation_ms << " ms\n"
                << "  Host to device: " << result.host_to_device_ms << " ms\n"
                << "  Kernel:         " << result.kernel_ms << " ms\n"
                << "  Device to host: " << result.device_to_host_ms << " ms\n"
                << "  Cleanup:        " << result.cleanup_ms << " ms\n"
                << "  Component sum:  " << component_sum_ms << " ms\n"
                << "  GPU total:      " << result.total_ms << " ms\n"
                << "  Executor time:  "
                << coordinator.last_execution_ms() << " ms\n";
        }
    }

    report.flush();

    if (!report) {
        std::cerr << "Failed while writing scheduling measurements\n";
        return 1;
    }

    if (!all_tasks_correct) {
        std::cerr << "One or more task validations failed.\n";
        return 1;
    }

    std::cout << "\nScheduler validation passed.\n"
              << "Measurement report: "
              << "build/scheduling_measurements.csv\n";

    return 0;
}
