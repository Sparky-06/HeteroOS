
#include "monitoring/resource_monitor.hpp"

#include <iostream>
#include <chrono>


bool ResourceMonitor::initialize() {

    // Initialize GPU monitoring independently.
    gpu_initialized = gpu_monitor.initialize();

    if (!gpu_initialized) {
        std::cerr << "GPU monitoring unavailable. "
                  << "Continuing with CPU and RAM.\n";
    }

    // CPU and memory monitoring do not require initialization.
    return true;
}


SystemSnapshot ResourceMonitor::collect() {

    SystemSnapshot snapshot;

    // Assign snapshot metadata.
    snapshot.sequence = ++sequence;

    snapshot.timestamp = std::chrono::steady_clock::now();


    // ----------------------------------
    // CPU COLLECTION
    // ----------------------------------

    snapshot.cpu_valid =
        cpu_monitor.collect(snapshot.cpu);


    // ----------------------------------
    // MEMORY COLLECTION
    // ----------------------------------

    snapshot.memory_valid =
        memory_monitor.read_memory_stats(snapshot.memory);


    // ----------------------------------
    // GPU COLLECTION
    // ----------------------------------

    if (gpu_initialized) {

        snapshot.gpu_valid =
            gpu_monitor.read_gpu_stats(snapshot.gpu);
    }


    return snapshot;
}
