
#include "monitoring/memory_monitor.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

bool MemoryMonitor::read_memory_stats(MemoryStats& output) {

    std::ifstream file("/proc/meminfo");

    if (!file.is_open()) {
        std::cerr << "Failed to open /proc/meminfo\n";
        return false;
    }

    MemoryStats new_stats;

    // Track whether every required field was found.
    bool total_found = false;
    bool free_found = false;
    bool available_found = false;
    bool buffers_found = false;
    bool cached_found = false;
    bool swap_cached_found = false;
    bool swap_total_found = false;
    bool swap_free_found = false;

    std::string line;

    while (std::getline(file, line)) {

        std::istringstream stream(line);

        std::string key;
        uint64_t value;

        if (!(stream >> key >> value)) {
            continue;
        }

        if (key == "MemTotal:") {
            new_stats.total = value;
            total_found = true;
        }
        else if (key == "MemFree:") {
            new_stats.free = value;
            free_found = true;
        }
        else if (key == "MemAvailable:") {
            new_stats.available = value;
            available_found = true;
        }
        else if (key == "Buffers:") {
            new_stats.buffers = value;
            buffers_found = true;
        }
        else if (key == "Cached:") {
            new_stats.cached = value;
            cached_found = true;
        }
        else if (key == "SwapCached:") {
            new_stats.swap_cached = value;
            swap_cached_found = true;
        }
        else if (key == "SwapTotal:") {
            new_stats.swap_total = value;
            swap_total_found = true;
        }
        else if (key == "SwapFree:") {
            new_stats.swap_free = value;
            swap_free_found = true;
        }
    }

    bool all_fields_found =
        total_found &&
        free_found &&
        available_found &&
        buffers_found &&
        cached_found &&
        swap_cached_found &&
        swap_total_found &&
        swap_free_found;

    if (!all_fields_found) {
        std::cerr << "Incomplete memory statistics\n";
        return false;
    }

    if (new_stats.total == 0 ||
        new_stats.free > new_stats.total ||
        new_stats.available > new_stats.total ||
        new_stats.swap_free > new_stats.swap_total) {

        std::cerr << "Invalid memory statistics\n";
        return false;
    }

    // Publish the snapshot only after successful validation.
    output = new_stats;

    return true;
}
