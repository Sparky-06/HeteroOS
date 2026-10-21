
#include "ui/terminal_dashboard.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>


// ----------------------------------
// UTILIZATION BAR
// ----------------------------------

void TerminalDashboard::print_bar(
    double percentage,
    int width
) {

    percentage = std::clamp(percentage, 0.0, 100.0);

    int filled = static_cast<int>(
        (percentage / 100.0) * width
    );

    std::cout << "[";

    for (int i = 0; i < width; i++) {

        if (i < filled) {
            std::cout << "#";
        }
        else {
            std::cout << "-";
        }
    }

    std::cout << "]";
}


// ----------------------------------
// CPU DISPLAY
// ----------------------------------

void TerminalDashboard::display_cpu(
    const CPUData& cpu,
    bool valid
) {

    std::cout << "\n========== CPU ==========\n\n";

    if (!valid) {

        std::cout << "CPU data warming up or unavailable.\n";

        return;
    }

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Total Usage : ";

    print_bar(cpu.total_usage);

    std::cout << " "
              << cpu.total_usage << "%\n\n";


    std::cout << "Per-Core Usage:\n\n";

    for (const auto& core : cpu.cores) {

        std::cout << std::setw(6)
                  << std::left
                  << core.name
                  << " ";

        print_bar(core.usage_percent, 20);

        std::cout << " "
                  << std::setw(7)
                  << std::right
                  << core.usage_percent
                  << "%\n";
    }
}


// ----------------------------------
// MEMORY DISPLAY
// ----------------------------------

void TerminalDashboard::display_memory(
    const MemoryStats& memory,
    bool valid
) {

    std::cout << "\n========== MEMORY ==========\n\n";

    if (!valid) {

        std::cout << "Memory data unavailable.\n";

        return;
    }

    // /proc/meminfo values are in KiB.
    constexpr double KIB_TO_GIB = 1024.0 * 1024.0;

    double total = memory.total / KIB_TO_GIB;

    double available = memory.available / KIB_TO_GIB;

    double used = total - available;

    double usage_percentage =
        (used / total) * 100.0;

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Total     : "
              << total << " GiB\n";

    std::cout << "Used      : "
              << used << " GiB\n";

    std::cout << "Available : "
              << available << " GiB\n\n";

    std::cout << "Usage     : ";

    print_bar(usage_percentage);

    std::cout << " "
              << usage_percentage << "%\n";

    double swap_total = memory.swap_total / KIB_TO_GIB;

    double swap_used =
        (memory.swap_total - memory.swap_free) / KIB_TO_GIB;

    std::cout << "\nSwap Used : "
              << swap_used << " / "
              << swap_total << " GiB\n";
}


// ----------------------------------
// GPU DISPLAY
// ----------------------------------

void TerminalDashboard::display_gpu(
    const GPUStats& gpu,
    bool valid
) {

    std::cout << "\n========== GPU ==========\n\n";

    if (!valid) {

        std::cout << "GPU telemetry unavailable.\n";

        return;
    }

    constexpr double BYTES_TO_GIB =
        1024.0 * 1024.0 * 1024.0;

    std::cout << "Device      : "
              << gpu.name << '\n';

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "\nGPU Usage   : ";

    print_bar(gpu.gpu_utilization);

    std::cout << " "
              << gpu.gpu_utilization << "%\n";

    std::cout << "Memory Util : ";

    print_bar(gpu.memory_utilization);

    std::cout << " "
              << gpu.memory_utilization << "%\n";


    std::cout << "\nVRAM Total  : "
              << gpu.memory_total / BYTES_TO_GIB
              << " GiB\n";

    std::cout << "VRAM Used   : "
              << gpu.memory_used / BYTES_TO_GIB
              << " GiB\n";

    std::cout << "VRAM Free   : "
              << gpu.memory_free / BYTES_TO_GIB
              << " GiB\n";


    std::cout << "\nTemperature : "
              << gpu.temperature << " C\n";

    std::cout << "Power Draw  : "
              << gpu.power_usage / 1000.0
              << " W\n";

    if (gpu.power_limit_available) {

        std::cout << "Power Limit : "
                  << gpu.power_limit / 1000.0
                  << " W\n";
    }
    else {

        std::cout << "Power Limit : Unavailable\n";
    }
}


// ----------------------------------
// MAIN RENDER FUNCTION
// ----------------------------------

void TerminalDashboard::render(
    const SystemSnapshot& snapshot
) {

    // Clear terminal and return cursor to top-left.
    std::cout << "\033[2J\033[H";

    std::cout << "============================================\n";
    std::cout << "              Hetero-OS MONITOR\n";
    std::cout << "============================================\n";

    std::cout << "Snapshot: "
              << snapshot.sequence << '\n';

    display_cpu(
        snapshot.cpu,
        snapshot.cpu_valid
    );

    display_memory(
        snapshot.memory,
        snapshot.memory_valid
    );

    display_gpu(
        snapshot.gpu,
        snapshot.gpu_valid
    );

    std::cout << "\n============================================\n";
    std::cout << "Refresh interval: 1 second\n";
    std::cout << "Press Ctrl+C to exit.\n";

    std::cout.flush();
}
