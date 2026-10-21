#include "scheduling/scheduler.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstdint>

namespace {

struct BenchmarkPoint {
    std::size_t dimension;
    double cpu_ms;
    double gpu_ms;
};

// Measured matrix multiplication profile.
// GPU costs include H2D + kernel + D2H, excluding allocations.
constexpr std::array<BenchmarkPoint, 2> MATRIX_PROFILE{{
    {256, 8.6552, 0.3054},
    {512, 99.4872, 1.3713}
}};

constexpr std::array<BenchmarkPoint, 2> VECTOR_PROFILE{{
    {65536, 0.0119, 0.2150},
    {1048576, 0.3481, 1.6700}
}};

template <std::size_t N>
std::optional<double> interpolate(
    const std::array<BenchmarkPoint, N>& profile,
    std::size_t dimension,
    bool use_gpu
) {
    // Never extrapolate beyond measured sizes.
    if (dimension < profile.front().dimension ||
        dimension > profile.back().dimension) {
        return std::nullopt;
    }

    for (const auto& point : profile) {
        if (dimension == point.dimension) {
            return use_gpu ? point.gpu_ms : point.cpu_ms;
        }
    }

    for (std::size_t i = 1; i < profile.size(); ++i) {
        if (dimension < profile[i].dimension) {
            const auto& lower = profile[i - 1];
            const auto& upper = profile[i];

            const double lower_cost =
                use_gpu ? lower.gpu_ms : lower.cpu_ms;

            const double upper_cost =
                use_gpu ? upper.gpu_ms : upper.cpu_ms;

            const double fraction =
                static_cast<double>(
                    dimension - lower.dimension
                ) /
                static_cast<double>(
                    upper.dimension - lower.dimension
                );

            return lower_cost +
                fraction * (upper_cost - lower_cost);
        }
    }

    return std::nullopt;
}

double clamp_percent(double value)
{
    return std::clamp(value, 0.0, 100.0);
}

// Heuristic, not a measured hardware law.
double cpu_pressure_factor(const SystemSnapshot& snapshot)
{
    if (!snapshot.cpu_valid) {
        return 1.0;
    }

    const double usage =
        clamp_percent(snapshot.cpu.total_usage);

    // At 0% usage: 1.0x.
    // At 100% usage: 1.5x.
    return 1.0 + 0.5 * usage / 100.0;
}

// Heuristic, not a measured hardware law.
double gpu_pressure_factor(const SystemSnapshot& snapshot)
{
    if (!snapshot.gpu_valid) {
        return 1.0;
    }

    const double usage = clamp_percent(
        static_cast<double>(snapshot.gpu.gpu_utilization)
    );

    // At 0% utilization: 1.0x.
    // At 100% utilization: 1.75x.
    return 1.0 + 0.75 * usage / 100.0;
}



std::optional<std::uint64_t> estimate_device_memory_bytes(
    const Task& task
) {
    const std::uint64_t dimension =
        static_cast<std::uint64_t>(task.dimension());

    std::uint64_t elements = dimension;

    if (task.type() == TaskType::MatrixMultiply) {
        if (dimension != 0 &&
            dimension >
                std::numeric_limits<std::uint64_t>::max() / dimension) {
            return std::nullopt;
        }

        elements = dimension * dimension;
    } else if (task.type() != TaskType::VectorAdd) {
        return std::nullopt;
    }

    // Current executors allocate three float buffers:
    // A, B and C.
    constexpr std::uint64_t bytes_per_element =
        3ULL * sizeof(float);

    if (elements >
        std::numeric_limits<std::uint64_t>::max() /
            bytes_per_element) {
        return std::nullopt;
    }

    return elements * bytes_per_element;
}

} // namespace

Scheduler::Scheduler(
    Profiler& profiler,
    std::uint64_t minimum_free_vram_bytes
)
    : profiler_(profiler),
      minimum_free_vram_bytes_(minimum_free_vram_bytes)
{
}


std::optional<double> Scheduler::estimate_cpu_ms(
    TaskType type,
    std::size_t dimension
) const {
    const auto historical = profiler_.estimate_ms(
        type,
        dimension,
        ExecutionTarget::CPU
    );

    if (historical) {
        return historical;
    }

    switch (type) {
        case TaskType::MatrixMultiply:
            return interpolate(MATRIX_PROFILE, dimension, false);

        case TaskType::VectorAdd:
            return interpolate(VECTOR_PROFILE, dimension, false);
    }

    return std::nullopt;
}

std::optional<double> Scheduler::estimate_gpu_ms(
    TaskType type,
    std::size_t dimension
) const {
    const auto historical = profiler_.estimate_ms(
        type,
        dimension,
        ExecutionTarget::GPU
    );

    if (historical) {
        return historical;
    }

    switch (type) {
        case TaskType::MatrixMultiply:
            return interpolate(MATRIX_PROFILE, dimension, true);

        case TaskType::VectorAdd:
            return interpolate(VECTOR_PROFILE, dimension, true);
    }

    return std::nullopt;
}


ScheduleDecision Scheduler::choose_target(
    const Task& task,
    const SystemSnapshot& snapshot,
    bool gpu_available
) const {
    ScheduleDecision decision;

    // 1. Establish the benchmark-derived costs first.
    decision.base_cpu_ms =
        estimate_cpu_ms(task.type(), task.dimension());

    decision.base_gpu_ms =
        estimate_gpu_ms(task.type(), task.dimension());

    if (decision.base_cpu_ms) {
        decision.estimated_cpu_ms =
            *decision.base_cpu_ms * cpu_pressure_factor(snapshot);
    }

    // Without a CPU profile, we cannot safely make a cost-based choice.
    if (!decision.base_cpu_ms) {
        decision.reason =
            "No CPU benchmark profile; conservative CPU fallback";
        decision.gpu_ineligibility_reason =
            "CPU profile unavailable";
        return decision;
    }

    // 2. Establish the GPU cost estimate independently of eligibility.
    if (!gpu_available) {
        decision.gpu_ineligibility_reason =
            "GPU executor unavailable";
        decision.reason =
            "GPU executor unavailable; using CPU";
        return decision;
    }

    if (!decision.base_gpu_ms) {
        decision.gpu_ineligibility_reason =
            "No GPU benchmark profile for this workload size";
        decision.reason =
            "GPU benchmark profile unavailable; using CPU";
        return decision;
    }

    if (!snapshot.gpu_valid) {
        decision.gpu_ineligibility_reason =
            "GPU telemetry unavailable";
        decision.reason =
            "GPU telemetry unavailable; conservative CPU fallback";
        return decision;
    }

    decision.estimated_gpu_ms =
        *decision.base_gpu_ms * gpu_pressure_factor(snapshot);

    // 3. Apply GPU feasibility checks.
    const auto required_bytes =
        estimate_device_memory_bytes(task);

    if (!required_bytes) {
        decision.gpu_ineligibility_reason =
            "Cannot safely estimate GPU memory requirement";
        decision.reason =
            "GPU memory requirement unknown; using CPU";
        return decision;
    }

    const std::uint64_t free_bytes =
        snapshot.gpu.memory_free;

    if (free_bytes <= minimum_free_vram_bytes_) {
        decision.gpu_ineligibility_reason =
            "Free VRAM is below the configured reserve";
        decision.reason =
            "GPU rejected by VRAM reserve policy; using CPU";
        return decision;
    }

    const std::uint64_t available_after_reserve =
        free_bytes - minimum_free_vram_bytes_;

    if (*required_bytes > available_after_reserve) {
        decision.gpu_ineligibility_reason =
            "Task buffers would violate the configured VRAM reserve";
        decision.reason =
            "GPU rejected by VRAM feasibility check; using CPU";
        return decision;
    }

    // 4. GPU passed all current eligibility checks.
    decision.gpu_eligible = true;

    // 5. Rank eligible options by estimated cost.
    if (*decision.estimated_gpu_ms <
        *decision.estimated_cpu_ms) {
        decision.target = ExecutionTarget::GPU;
        decision.reason =
            "GPU eligible and its adjusted estimated cost is lower";
    } else {
        decision.target = ExecutionTarget::CPU;
        decision.reason =
            "CPU has the lower or equal adjusted estimated cost";
    }

    return decision;
}