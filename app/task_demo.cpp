
#include "task/task.hpp"

#include <iostream>
#include <vector>

int main()
{
    Task task(
        TaskType::VectorAdd,
        4,
        std::vector<float>{1, 2, 3, 4},
        std::vector<float>{10, 20, 30, 40}
    );

    if (!task.assign_id(1)) {
        std::cerr << "Failed to assign task ID\n";
        return 1;
    }

    std::cout << "Task ID: " << task.id() << '\n';
    std::cout << "Type: " << to_string(task.type()) << '\n';
    std::cout << "State: " << to_string(task.state()) << '\n';

    if (!task.enqueue()) {
        std::cerr << "Failed to enqueue task\n";
        return 1;
    }

    if (!task.set_execution_target(ExecutionTarget::CPU)) {
        std::cerr << "Failed to assign target\n";
        return 1;
    }

    if (!task.start()) {
        std::cerr << "Failed to start task\n";
        return 1;
    }

    // Simulate execution. The actual executor integration
    // will be implemented in a later step.
    for (std::size_t i = 0; i < task.dimension(); ++i) {
        task.output()[i] =
            task.input_a()[i] + task.input_b()[i];
    }

    if (!task.complete()) {
        std::cerr << "Failed to complete task\n";
        return 1;
    }

    std::cout << "Target: " << to_string(task.target()) << '\n';
    std::cout << "Final state: " << to_string(task.state()) << '\n';
    std::cout << "Data storage: "
              << task.memory_required_bytes() << " bytes\n";

    std::cout << "Output: ";

    for (float value : task.output()) {
        std::cout << value << ' ';
    }

    std::cout << '\n';

    // A terminal task cannot be started again.
    if (task.start()) {
        std::cerr << "Invalid transition was accepted\n";
        return 1;
    }

    std::cout << "Lifecycle validation passed.\n";

    return 0;
}
