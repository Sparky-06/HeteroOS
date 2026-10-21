#include "monitoring/gpu_monitor.hpp"

#include <iostream>


// Destructor: release NVML resources.
GPUMonitor::~GPUMonitor() {

    shutdown();
}


// Initialize NVML and obtain the GPU handle.
bool GPUMonitor::initialize() {

    if (initialized) {
        return true;
    }

    nvmlReturn_t result = nvmlInit_v2();

    if (result != NVML_SUCCESS) {

        std::cerr << "NVML initialization failed: "
                  << nvmlErrorString(result) << '\n';

        return false;
    }

    initialized = true;

    unsigned int device_count = 0;

    result = nvmlDeviceGetCount_v2(&device_count);

    if (result != NVML_SUCCESS || device_count == 0) {

        std::cerr << "No NVIDIA GPU detected\n";

        shutdown();

        return false;
    }

    // Use the first detected NVIDIA GPU.
    result = nvmlDeviceGetHandleByIndex_v2(0, &device);

    if (result != NVML_SUCCESS) {

        std::cerr << "Failed to get GPU handle: "
                  << nvmlErrorString(result) << '\n';

        shutdown();

        return false;
    }

    return true;
}


// Collect GPU statistics.
bool GPUMonitor::read_gpu_stats(GPUStats& output) {

    if (!initialized) {

        std::cerr << "GPU monitor is not initialized\n";

        return false;
    }

    // Temporary object prevents partially collected data
    // from being published.
    GPUStats new_stats;

    nvmlReturn_t result;


    // ----------------------------------
    // GPU NAME
    // ----------------------------------

    char name[NVML_DEVICE_NAME_BUFFER_SIZE];

    result = nvmlDeviceGetName(
        device,
        name,
        sizeof(name)
    );

    if (result != NVML_SUCCESS) {

        std::cerr << "Failed to read GPU name: "
                  << nvmlErrorString(result) << '\n';

        return false;
    }

    new_stats.name = name;


    // ----------------------------------
    // GPU UTILIZATION
    // ----------------------------------

    nvmlUtilization_t utilization{};

    result = nvmlDeviceGetUtilizationRates(
        device,
        &utilization
    );

    if (result != NVML_SUCCESS) {

        std::cerr << "Failed to read GPU utilization: "
                  << nvmlErrorString(result) << '\n';

        return false;
    }

    new_stats.gpu_utilization = utilization.gpu;

    new_stats.memory_utilization = utilization.memory;


    // ----------------------------------
    // GPU MEMORY / VRAM
    // ----------------------------------

    nvmlMemory_t memory{};

    result = nvmlDeviceGetMemoryInfo(
        device,
        &memory
    );

    if (result != NVML_SUCCESS) {

        std::cerr << "Failed to read GPU memory: "
                  << nvmlErrorString(result) << '\n';

        return false;
    }

    new_stats.memory_total = memory.total;

    new_stats.memory_used = memory.used;

    new_stats.memory_free = memory.free;


    // ----------------------------------
    // GPU TEMPERATURE
    // ----------------------------------

    unsigned int temperature = 0;

    result = nvmlDeviceGetTemperature(
        device,
        NVML_TEMPERATURE_GPU,
        &temperature
    );

    if (result != NVML_SUCCESS) {

        std::cerr << "Failed to read GPU temperature: "
                << nvmlErrorString(result) << '\n';

        return false;
    }

    new_stats.temperature = temperature;


    // ----------------------------------
    // GPU POWER CONSUMPTION
    // ----------------------------------

    unsigned int power = 0;

    result = nvmlDeviceGetPowerUsage(
        device,
        &power
    );

    if (result != NVML_SUCCESS) {

        std::cerr << "Failed to read GPU power: "
                  << nvmlErrorString(result) << '\n';

        return false;
    }

    new_stats.power_usage = power;


    // ----------------------------------
    // GPU POWER LIMIT
    // ----------------------------------

    unsigned int power_limit = 0;

    result = nvmlDeviceGetPowerManagementLimit(
        device,
        &power_limit
    );

    if (result == NVML_SUCCESS) {

        new_stats.power_limit = power_limit;

        new_stats.power_limit_available = true;
    }

    else {

        // Power limit is optional.
        // Unsupported queries must not fail collection.

        new_stats.power_limit_available = false;
    }


    // ----------------------------------
    // PUBLISH SNAPSHOT
    // ----------------------------------

    output = std::move(new_stats);

    return true;
}


// Shut down NVML.
void GPUMonitor::shutdown() {

    if (initialized) {

        nvmlShutdown();

        initialized = false;
    }
}
