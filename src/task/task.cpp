
#include "task/task.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace {

std::size_t expected_elements(
    TaskType type,
    std::size_t dimension
) {
    if (dimension == 0) {
        throw std::invalid_argument(
            "Task dimension must be greater than zero"
        );
    }

    if (type == TaskType::MatrixMultiply) {
        if (dimension >
            std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::length_error(
                "Matrix dimensions overflow size_t"
            );
        }

        return dimension * dimension;
    }

    if (type == TaskType::VectorAdd) {
        return dimension;
    }

    throw std::invalid_argument("Unsupported task type");
}

} // namespace

Task::Task(
    TaskType type,
    std::size_t dimension,
    std::vector<float> A,
    std::vector<float> B
)
    : type_(type),
      dimension_(dimension),
      A_(std::move(A)),
      B_(std::move(B)),
      C_(expected_elements(type, dimension), 0.0f)
{
    const std::size_t expected =
        expected_elements(type_, dimension_);

    if (A_.size() != expected || B_.size() != expected) {
        throw std::invalid_argument(
            "Input buffer sizes do not match the task dimensions"
        );
    }
}

bool Task::assign_id(std::uint64_t id)
{
    if (state_ != TaskState::Created ||
        id_ != 0 ||
        id == 0) {
        return false;
    }

    id_ = id;
    return true;
}

bool Task::enqueue()
{
    if (id_ == 0) {
        return false;
    }

    return transition_to(TaskState::Queued);
}

bool Task::start()
{
    if (target_ == ExecutionTarget::Unassigned) {
        return false;
    }

    return transition_to(TaskState::Running);
}

bool Task::complete()
{
    return transition_to(TaskState::Completed);
}

bool Task::fail()
{
    return transition_to(TaskState::Failed);
}

bool Task::set_execution_target(ExecutionTarget target)
{
    if (state_ != TaskState::Queued ||
        target_ != ExecutionTarget::Unassigned ||
        target == ExecutionTarget::Unassigned) {
        return false;
    }

    target_ = target;
    return true;
}

bool Task::transition_to(TaskState next)
{
    switch (state_) {
        case TaskState::Created:
            if (next == TaskState::Queued ||
                next == TaskState::Failed) {
                state_ = next;
                return true;
            }
            break;

        case TaskState::Queued:
            if (next == TaskState::Running ||
                next == TaskState::Failed) {
                state_ = next;
                return true;
            }
            break;

        case TaskState::Running:
            if (next == TaskState::Completed ||
                next == TaskState::Failed) {
                state_ = next;
                return true;
            }
            break;

        case TaskState::Completed:
        case TaskState::Failed:
            // Terminal states cannot be exited.
            break;
    }

    return false;
}

std::uint64_t Task::id() const
{
    return id_;
}

TaskType Task::type() const
{
    return type_;
}

TaskState Task::state() const
{
    return state_;
}

ExecutionTarget Task::target() const
{
    return target_;
}

std::size_t Task::dimension() const
{
    return dimension_;
}

std::size_t Task::memory_required_bytes() const
{
    const std::size_t elements = A_.size();

    constexpr std::size_t bytes_per_element =
        3 * sizeof(float);

    if (elements >
        std::numeric_limits<std::size_t>::max() /
            bytes_per_element) {
        throw std::length_error(
            "Task memory requirement overflows size_t"
        );
    }

    return elements * bytes_per_element;
}

const std::vector<float>& Task::input_a() const
{
    return A_;
}

const std::vector<float>& Task::input_b() const
{
    return B_;
}

std::vector<float>& Task::output()
{
    return C_;
}

const std::vector<float>& Task::output() const
{
    return C_;
}

const char* to_string(TaskType type)
{
    switch (type) {
        case TaskType::MatrixMultiply:
            return "MatrixMultiply";
        case TaskType::VectorAdd:
            return "VectorAdd";
    }

    return "Unknown";
}

const char* to_string(TaskState state)
{
    switch (state) {
        case TaskState::Created:
            return "Created";
        case TaskState::Queued:
            return "Queued";
        case TaskState::Running:
            return "Running";
        case TaskState::Completed:
            return "Completed";
        case TaskState::Failed:
            return "Failed";
    }

    return "Unknown";
}

const char* to_string(ExecutionTarget target)
{
    switch (target) {
        case ExecutionTarget::Unassigned:
            return "Unassigned";
        case ExecutionTarget::CPU:
            return "CPU";
        case ExecutionTarget::GPU:
            return "GPU";
    }

    return "Unknown";
}
