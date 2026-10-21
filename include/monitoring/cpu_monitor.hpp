#ifndef CPU_MONITOR_HPP
#define CPU_MONITOR_HPP

#include "monitoring/resource_types.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct CPUStats {
    uint64_t user = 0;
    uint64_t nice = 0;
    uint64_t system = 0;
    uint64_t idle = 0;
    uint64_t iowait = 0;
    uint64_t irq = 0;
    uint64_t softirq = 0;
    uint64_t steal = 0;
};

struct CPUInfo {
    std::string name;
    CPUStats stats;
};

class CPUMonitor {

private:
    std::vector<CPUInfo> previous_stats;
    bool initialized = false;

    bool read_cpu_stats(std::vector<CPUInfo>& result);

    double calculate_usage(
        const CPUStats& previous,
        const CPUStats& current
    ) const;

public:
    bool collect(CPUData& output);
};

#endif