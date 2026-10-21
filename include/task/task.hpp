
#ifndef HETEROOS_TASK_HPP
#define HETEROOS_TASK_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

enum class TaskType {
    MatrixMultiply,
    VectorAdd
};

enum class TaskState {
    Created,
    Queued,
    Running,
    Completed,
    Failed
};

enum class ExecutionTarget {
    Unassigned,
    CPU,
    GPU
};

class Task {
public:
    Task(
        TaskType type,
        std::size_t dimension,
        std::vector<float> A,
        std::vector<float> B
    );

    // Task identity is assigned by the future TaskManager.
    bool assign_id(std::uint64_t id);

    // Explicit lifecycle transitions.
    bool enqueue();
    bool start();
    bool complete();
    bool fail();

    // The scheduler will assign this before execution.
    bool set_execution_target(ExecutionTarget target);

    // Read-only metadata.
    std::uint64_t id() const;
    TaskType type() const;
    TaskState state() const;
    ExecutionTarget target() const;
    std::size_t dimension() const;

    // Approximate storage for A, B and C, in bytes.
    std::size_t memory_required_bytes() const;

    // Input buffers are immutable through this interface.
    const std::vector<float>& input_a() const;
    const std::vector<float>& input_b() const;

    // The executor writes the output buffer.
    std::vector<float>& output();
    const std::vector<float>& output() const;

private:
    bool transition_to(TaskState next);

    std::uint64_t id_ = 0;
    TaskType type_;
    TaskState state_ = TaskState::Created;
    ExecutionTarget target_ = ExecutionTarget::Unassigned;

    std::size_t dimension_;

    std::vector<float> A_;
    std::vector<float> B_;
    std::vector<float> C_;
};

const char* to_string(TaskType type);
const char* to_string(TaskState state);
const char* to_string(ExecutionTarget target);

#endif
