
#ifndef HETEROOS_PROFILER_HPP
#define HETEROOS_PROFILER_HPP

#include "task/task.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <vector>
#include <string>

struct ExecutionRecord {
    TaskType task_type;
    std::size_t dimension;
    ExecutionTarget target;
    double execution_ms;
    bool success;
};

struct WorkloadKey {
    TaskType task_type;
    std::size_t dimension;
    ExecutionTarget target;

    bool operator<(const WorkloadKey& other) const
    {
        if (task_type != other.task_type) {
            return task_type < other.task_type;
        }

        if (dimension != other.dimension) {
            return dimension < other.dimension;
        }

        return target < other.target;
    }
};

struct WorkloadProfile {
    std::size_t sample_count = 0;
    std::vector<double> execution_times_ms;

    std::optional<double> median_ms() const;
};

class Profiler {
public:
    void record(const ExecutionRecord& record);

    std::optional<double> estimate_ms(
        TaskType task_type,
        std::size_t dimension,
        ExecutionTarget target,
        std::size_t minimum_samples = 3
    ) const;

    std::size_t sample_count(
        TaskType task_type,
        std::size_t dimension,
        ExecutionTarget target
    ) const;

    bool save_csv(const std::string& path) const;
    bool load_csv(const std::string& path);

private:
    mutable std::mutex mutex_;
    std::map<WorkloadKey, WorkloadProfile> profiles_;
};

#endif
