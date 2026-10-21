#ifndef SYSTEM_SNAPSHOT_HPP
#define SYSTEM_SNAPSHOT_HPP

#include "monitoring/resource_types.hpp"

#include <chrono>
#include <cstdint>

struct SystemSnapshot {

    uint64_t sequence = 0;

    std::chrono::steady_clock::time_point timestamp;

    CPUData cpu;
    MemoryStats memory;
    GPUStats gpu;

    bool cpu_valid = false;
    bool memory_valid = false;
    bool gpu_valid = false;
};

#endif