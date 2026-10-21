
#ifndef TERMINAL_DASHBOARD_HPP
#define TERMINAL_DASHBOARD_HPP

#include "monitoring/system_snapshot.hpp"

class TerminalDashboard {

private:

    void print_bar(double percentage, int width = 30);

    void display_cpu(const CPUData& cpu, bool valid);

    void display_memory(const MemoryStats& memory, bool valid);

    void display_gpu(const GPUStats& gpu, bool valid);

public:

    void render(const SystemSnapshot& snapshot);

};

#endif
