#ifndef MEMORY_MONITOR_HPP
#define MEMORY_MONITOR_HPP

#include "monitoring/resource_types.hpp"

class MemoryMonitor {

public:
    bool read_memory_stats(MemoryStats& output);

};

#endif