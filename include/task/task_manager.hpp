#ifndef HETEROOS_TASK_MANAGER_HPP
#define HETEROOS_TASK_MANAGER_HPP

#include "task/task.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>

class TaskManager {
public:
    // Takes ownership of a newly created task.
    // Returns its ID, or nullopt if submission is invalid.
    std::optional<std::uint64_t> submit(
        std::unique_ptr<Task> task
    );

    // Removes the next valid ID from the pending queue.
    // The task remains in the registry and in Queued state.
    Task* take_next_task();

    // Retrieve a task, including completed or failed tasks.
    Task* get_task(std::uint64_t id);
    const Task* get_task(std::uint64_t id) const;

    std::size_t queued_count() const;
    std::size_t task_count() const;
    bool empty() const;

private:
    std::uint64_t next_id_ = 1;

    std::unordered_map<std::uint64_t, std::unique_ptr<Task>> tasks_;
    std::deque<std::uint64_t> pending_queue_;
};

#endif