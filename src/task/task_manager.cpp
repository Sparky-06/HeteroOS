#include "task/task_manager.hpp"

#include <limits>
#include <utility>

std::optional<std::uint64_t> TaskManager::submit(
    std::unique_ptr<Task> task
)
{
    if (!task ||
        task->state() != TaskState::Created ||
        task->id() != 0 ||
        next_id_ == 0) {
        return std::nullopt;
    }

    const std::uint64_t id = next_id_;

    if (!task->assign_id(id)) {
        return std::nullopt;
    }

    if (!task->enqueue()) {
        return std::nullopt;
    }

    // The registry owns the task. The queue contains only its ID.
    auto [it, inserted] =
        tasks_.emplace(id, std::move(task));

    if (!inserted) {
        return std::nullopt;
    }

    try {
        pending_queue_.push_back(id);
    } catch (...) {
        // Preserve consistency if queue allocation fails.
        tasks_.erase(it);
        throw;
    }

    // Unsigned wraparound produces zero, which marks ID exhaustion.
    ++next_id_;

    return id;
}

Task* TaskManager::take_next_task()
{
    while (!pending_queue_.empty()) {
        const std::uint64_t id = pending_queue_.front();
        pending_queue_.pop_front();

        auto it = tasks_.find(id);

        if (it == tasks_.end()) {
            continue;
        }

        Task* task = it->second.get();

        // Ignore stale queue entries if a task has changed state.
        if (task->state() != TaskState::Queued) {
            continue;
        }

        return task;
    }

    return nullptr;
}

Task* TaskManager::get_task(std::uint64_t id)
{
    auto it = tasks_.find(id);

    if (it == tasks_.end()) {
        return nullptr;
    }

    return it->second.get();
}

const Task* TaskManager::get_task(std::uint64_t id) const
{
    auto it = tasks_.find(id);

    if (it == tasks_.end()) {
        return nullptr;
    }

    return it->second.get();
}

std::size_t TaskManager::queued_count() const
{
    return pending_queue_.size();
}

std::size_t TaskManager::task_count() const
{
    return tasks_.size();
}

bool TaskManager::empty() const
{
    return pending_queue_.empty();
}