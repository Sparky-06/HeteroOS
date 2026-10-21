#ifndef RESOURCE_MONITOR_HPP
#define RESOURCE_MONITOR_HPP

#include "monitoring/system_snapshot.hpp"

#include "monitoring/cpu_monitor.hpp"
#include "monitoring/memory_monitor.hpp"
#include "monitoring/gpu_monitor.hpp"

#include <cstdint>

class ResourceMonitor {

private:

    CPUMonitor cpu_monitor;
    MemoryMonitor memory_monitor;
    GPUMonitor gpu_monitor;

    uint64_t sequence = 0;
    bool gpu_initialized = false;

public:

    ResourceMonitor() = default;

    ~ResourceMonitor() = default;

    ResourceMonitor(const ResourceMonitor&) = delete;
    ResourceMonitor& operator=(const ResourceMonitor&) = delete;

    bool initialize();

    SystemSnapshot collect();

};

#endif