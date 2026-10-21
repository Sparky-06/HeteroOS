#ifndef GPU_MONITOR_HPP
#define GPU_MONITOR_HPP

#include "monitoring/resource_types.hpp"

#include <nvml.h>

class GPUMonitor {

private:
    nvmlDevice_t device{};
    bool initialized = false;

public:
    GPUMonitor() = default;
    ~GPUMonitor();

    GPUMonitor(const GPUMonitor&) = delete;
    GPUMonitor& operator=(const GPUMonitor&) = delete;

    bool initialize();

    bool read_gpu_stats(GPUStats& output);

    void shutdown();
};

#endif