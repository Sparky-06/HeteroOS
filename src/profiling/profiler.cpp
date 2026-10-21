
#include "profiling/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <exception>
#include <utility>


std::optional<double> WorkloadProfile::median_ms() const
{
    if (execution_times_ms.empty()) {
        return std::nullopt;
    }

    std::vector<double> sorted = execution_times_ms;

    std::sort(sorted.begin(), sorted.end());

    const std::size_t middle = sorted.size() / 2;

    if (sorted.size() % 2 != 0) {
        return sorted[middle];
    }

    return (sorted[middle - 1] + sorted[middle]) / 2.0;
}

void Profiler::record(const ExecutionRecord& record)
{
    if (!record.success
        || !std::isfinite(record.execution_ms)
        || record.execution_ms <= 0.0
        || record.target == ExecutionTarget::Unassigned) {
        return;
    }

    const WorkloadKey key{
        record.task_type,
        record.dimension,
        record.target
    };

    std::lock_guard<std::mutex> lock(mutex_);

    auto& profile = profiles_[key];

    profile.execution_times_ms.push_back(record.execution_ms);
    profile.sample_count = profile.execution_times_ms.size();
}

std::optional<double> Profiler::estimate_ms(
    TaskType task_type,
    std::size_t dimension,
    ExecutionTarget target,
    std::size_t minimum_samples
) const {
    const WorkloadKey key{
        task_type,
        dimension,
        target
    };

    std::lock_guard<std::mutex> lock(mutex_);

    const auto it = profiles_.find(key);

    if (it == profiles_.end()
        || it->second.sample_count < minimum_samples) {
        return std::nullopt;
    }

    return it->second.median_ms();
}

std::size_t Profiler::sample_count(
    TaskType task_type,
    std::size_t dimension,
    ExecutionTarget target
) const {
    const WorkloadKey key{
        task_type,
        dimension,
        target
    };

    std::lock_guard<std::mutex> lock(mutex_);

    const auto it = profiles_.find(key);

    if (it == profiles_.end()) {
        return 0;
    }

    return it->second.sample_count;
}



bool Profiler::save_csv(const std::string& path) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream file(path, std::ios::trunc);

    if (!file) {
        return false;
    }

    file << "task_type,dimension,target,execution_ms\n";

    for (const auto& [key, profile] : profiles_) {
        for (const double time_ms : profile.execution_times_ms) {
            file << to_string(key.task_type) << ','
                 << key.dimension << ','
                 << to_string(key.target) << ','
                 << time_ms << '\n';
        }
    }

    return static_cast<bool>(file);
}

bool Profiler::load_csv(const std::string& path)
{
    std::ifstream file(path);

    if (!file) {
        return false;
    }

    std::string line;

    if (!std::getline(file, line) ||
        line != "task_type,dimension,target,execution_ms") {
        return false;
    }

    Profiler loaded;

    while (std::getline(file, line)) {
        std::istringstream row(line);

        std::string type_text;
        std::string dimension_text;
        std::string target_text;
        std::string time_text;
        std::string extra;

        if (!std::getline(row, type_text, ',') ||
            !std::getline(row, dimension_text, ',') ||
            !std::getline(row, target_text, ',') ||
            !std::getline(row, time_text, ',') ||
            std::getline(row, extra, ',')) {
            return false;
        }

        TaskType type;
        ExecutionTarget target;

        if (type_text == to_string(TaskType::VectorAdd)) {
            type = TaskType::VectorAdd;
        } else if (type_text == to_string(TaskType::MatrixMultiply)) {
            type = TaskType::MatrixMultiply;
        } else {
            return false;
        }

        if (target_text == to_string(ExecutionTarget::CPU)) {
            target = ExecutionTarget::CPU;
        } else if (target_text == to_string(ExecutionTarget::GPU)) {
            target = ExecutionTarget::GPU;
        } else {
            return false;
        }

        try {
            std::size_t dimension_chars = 0;
            std::size_t time_chars = 0;

            const auto dimension_value =
                std::stoull(dimension_text, &dimension_chars);

            const double execution_ms =
                std::stod(time_text, &time_chars);

            if (dimension_chars != dimension_text.size() ||
                time_chars != time_text.size() ||
                dimension_value == 0) {
                return false;
            }

            const auto dimension =
                static_cast<std::size_t>(dimension_value);

            if (static_cast<unsigned long long>(dimension) !=
                dimension_value) {
                return false;
            }

            loaded.record({
                type,
                dimension,
                target,
                execution_ms,
                true
            });

            if (loaded.sample_count(type, dimension, target) == 0) {
                return false;
            }
        } catch (const std::exception&) {
            return false;
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    profiles_ = std::move(loaded.profiles_);

    return true;
}
