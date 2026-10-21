
#include "monitoring/cpu_monitor.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>

// Calculate the total CPU time.
static uint64_t total_time(const CPUStats& s) {

    return s.user +
           s.nice +
           s.system +
           s.idle +
           s.iowait +
           s.irq +
           s.softirq +
           s.steal;
}

// Calculate idle CPU time.
static uint64_t idle_time(const CPUStats& s) {

    return s.idle + s.iowait;
}


// Read CPU statistics from /proc/stat.
bool CPUMonitor::read_cpu_stats(
    std::vector<CPUInfo>& result
) {

    std::ifstream file("/proc/stat");

    if (!file.is_open()) {
        std::cerr << "Failed to open /proc/stat\n";
        return false;
    }

    std::vector<CPUInfo> new_stats;

    std::string line;

    while (std::getline(file, line)) {

        std::istringstream stream(line);

        CPUInfo info;

        if (!(stream >> info.name)) {
            continue;
        }

        // Only process CPU entries.
        if (info.name.rfind("cpu", 0) != 0) {
            continue;
        }

        // Read and validate all eight required counters.
        if (!(stream >> info.stats.user
                     >> info.stats.nice
                     >> info.stats.system
                     >> info.stats.idle
                     >> info.stats.iowait
                     >> info.stats.irq
                     >> info.stats.softirq
                     >> info.stats.steal)) {

            std::cerr << "Invalid CPU statistics for "
                      << info.name << '\n';

            return false;
        }

        new_stats.push_back(info);
    }

    // The aggregate CPU entry must exist.
    if (new_stats.empty() || new_stats.front().name != "cpu") {

        std::cerr << "Aggregate CPU statistics unavailable\n";
        return false;
    }

    result = std::move(new_stats);

    return true;
}


// Calculate CPU utilization between two snapshots.
double CPUMonitor::calculate_usage(
    const CPUStats& previous,
    const CPUStats& current
) const {

    uint64_t previous_total = total_time(previous);
    uint64_t current_total = total_time(current);

    uint64_t previous_idle = idle_time(previous);
    uint64_t current_idle = idle_time(current);

    // Guard against invalid or decreasing counters.
    if (current_total <= previous_total ||
        current_idle < previous_idle) {

        return 0.0;
    }

    uint64_t total_delta =
        current_total - previous_total;

    uint64_t idle_delta =
        current_idle - previous_idle;

    if (total_delta == 0 || idle_delta > total_delta) {
        return 0.0;
    }

    return 100.0 *
        (1.0 -
        static_cast<double>(idle_delta) / total_delta);
}


// Collect CPU statistics.
bool CPUMonitor::collect(CPUData& output) {

    std::vector<CPUInfo> current_stats;

    if (!read_cpu_stats(current_stats)) {
        return false;
    }

    // First collection establishes the baseline.
    if (!initialized) {

        previous_stats = current_stats;
        initialized = true;

        return false;
    }

    // Map previous CPU statistics by CPU name.
    std::unordered_map<std::string, CPUStats> previous_map;

    for (const auto& cpu : previous_stats) {
        previous_map[cpu.name] = cpu.stats;
    }

    CPUData new_output;

    // Find the previous aggregate CPU snapshot.
    auto aggregate_previous = previous_map.find("cpu");

    if (aggregate_previous == previous_map.end()) {
        previous_stats = current_stats;
        return false;
    }

    // Calculate aggregate CPU utilization.
    new_output.total_usage = calculate_usage(
        aggregate_previous->second,
        current_stats.front().stats
    );

    // Calculate individual logical CPU utilization.
    for (const auto& cpu : current_stats) {

        // Skip aggregate CPU entry.
        if (cpu.name == "cpu") {
            continue;
        }

        auto previous = previous_map.find(cpu.name);

        // Skip CPUs without a matching previous snapshot.
        if (previous == previous_map.end()) {
            continue;
        }

        CPUCoreUsage core;

        core.name = cpu.name;

        core.usage_percent = calculate_usage(
            previous->second,
            cpu.stats
        );

        new_output.cores.push_back(core);
    }

    // Publish the collected data.
    output = std::move(new_output);

    // Current snapshot becomes the next baseline.
    previous_stats = std::move(current_stats);

    return true;
}
