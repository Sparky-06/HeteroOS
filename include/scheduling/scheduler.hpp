#ifndef HETEROOS_SCHEDULER_HPP
#define HETEROOS_SCHEDULER_HPP

#include "monitoring/system_snapshot.hpp"
#include "task/task.hpp"
#include "profiling/profiler.hpp"

#include <optional>
#include <string>
#include <cstdint>


struct ScheduleDecision {
    ExecutionTarget target = ExecutionTarget::CPU;

    // Original benchmark-derived costs.
    std::optional<double> base_cpu_ms;
    std::optional<double> base_gpu_ms;

    // Costs adjusted for current resource pressure.
    std::optional<double> estimated_cpu_ms;
    std::optional<double> estimated_gpu_ms;

    // Feasibility is separate from performance.
    bool gpu_eligible = false;
    std::string gpu_ineligibility_reason;

    // Explanation of the final target selection.
    std::string reason;
};


class Scheduler {
public:
    explicit Scheduler(
        Profiler& profiler,
        std::uint64_t minimum_free_vram_bytes =
            512ULL * 1024ULL * 1024ULL
    );

    ScheduleDecision choose_target(
        const Task& task,
        const SystemSnapshot& snapshot,
        bool gpu_available
    ) const;

private:
    Profiler& profiler_;
    std::uint64_t minimum_free_vram_bytes_;

    std::optional<double> estimate_cpu_ms(
        TaskType type,
        std::size_t dimension
    ) const;

    std::optional<double> estimate_gpu_ms(
        TaskType type,
        std::size_t dimension
    ) const;
};


#endif