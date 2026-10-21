
#include "monitoring/resource_monitor.hpp"
#include "ui/terminal_dashboard.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>


volatile std::sig_atomic_t running = 1;


void handle_signal(int) {

    running = 0;
}


int main() {

    std::signal(SIGINT, handle_signal);

    ResourceMonitor monitor;

    if (!monitor.initialize()) {

        std::cerr << "Failed to initialize HeteroOS\n";

        return 1;
    }

    TerminalDashboard dashboard;

    while (running) {

        SystemSnapshot snapshot = monitor.collect();

        dashboard.render(snapshot);

        std::this_thread::sleep_for(
            std::chrono::seconds(1)
        );
    }

    std::cout << "\nHeteroOS Monitor stopped.\n";

    return 0;
}
