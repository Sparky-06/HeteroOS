#ifndef RESOURCE_TYPES_HPP
#define RESOURCE_TYPES_HPP

#include <cstdint>
#include <string>
#include <vector>

struct CPUCoreUsage {
    std::string name;
    double usage_percent = 0.0;
};

struct CPUData {
    double total_usage = 0.0;
    std::vector<CPUCoreUsage> cores;
};

struct MemoryStats {
    uint64_t total = 0;
    uint64_t free = 0;
    uint64_t available = 0;
    uint64_t buffers = 0;
    uint64_t cached = 0;
    uint64_t swap_cached = 0;
    uint64_t swap_total = 0;
    uint64_t swap_free = 0;
};

struct GPUStats {
    std::string name;

    uint32_t gpu_utilization = 0;
    uint32_t memory_utilization = 0;

    uint64_t memory_total = 0;
    uint64_t memory_used = 0;
    uint64_t memory_free = 0;

    uint32_t temperature = 0;

    uint32_t power_usage = 0;
    uint32_t power_limit = 0;

    bool power_limit_available = false;
};

#endif