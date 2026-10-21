#include "task/task_manager.hpp"

#include <iostream>
#include <memory>
#include <vector>

int main()
{
    TaskManager manager;

    auto task = std::make_unique<Task>(
        TaskType::VectorAdd,
        4,
        std::vector<float>{1, 2, 3, 4},
        std::vector<float>{10, 20, 30, 40}
    );

    const auto submitted_id = manager.submit(std::move(task));

    if (!submitted_id) {
        std::cerr << "Task submission failed\n";
        return 1;
    }

    const std::uint64_t id = *submitted_id;

    std::cout << "Submitted task ID: " << id << '\n';
    std::cout << "Tasks retained: " << manager.task_count() << '\n';
    std::cout << "Tasks queued: " << manager.queued_count() << '\n';

    Task* next = manager.take_next_task();

    if (next == nullptr || next->id() != id) {
        std::cerr << "Failed to dequeue task\n";
        return 1;
    }

    if (!next->set_execution_target(ExecutionTarget::CPU) ||
        !next->start()) {
        std::cerr << "Failed to start task\n";
        return 1;
    }

    // Temporary execution simulation.
    // Actual executor dispatch comes in a later phase.
    if (next->type() == TaskType::VectorAdd) {
        for (std::size_t i = 0; i < next->dimension(); ++i) {
            next->output()[i] =
                next->input_a()[i] + next->input_b()[i];
        }
    } else {
        std::cerr << "Unsupported demo task type\n";
        next->fail();
        return 1;
    }

    if (!next->complete()) {
        std::cerr << "Failed to complete task\n";
        return 1;
    }

    // Look it up again through the manager.
    const Task* finished = manager.get_task(id);

    if (finished == nullptr ||
        finished->state() != TaskState::Completed) {
        std::cerr << "Could not retrieve completed task\n";
        return 1;
    }

    std::cout << "Task state: "
              << to_string(finished->state()) << '\n';

    std::cout << "Execution target: "
              << to_string(finished->target()) << '\n';

    std::cout << "Output: ";

    for (float value : finished->output()) {
        std::cout << value << ' ';
    }

    std::cout << "\nPending queue empty: "
              << (manager.empty() ? "yes" : "no") << '\n';

    std::cout << "Task manager validation passed.\n";

    return 0;
}