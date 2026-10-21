
#include "profiling/profiler.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    Profiler profiler;

    const auto target = ExecutionTarget::CPU;

    assert(!profiler.estimate_ms(
        TaskType::VectorAdd, 65536, target));

    profiler.record({
        TaskType::VectorAdd, 65536, target, 0.010, true
    });
    profiler.record({
        TaskType::VectorAdd, 65536, target, 0.012, true
    });
    profiler.record({
        TaskType::VectorAdd, 65536, target, 0.011, true
    });

    const auto estimate = profiler.estimate_ms(
        TaskType::VectorAdd, 65536, target);

    assert(estimate.has_value());
    assert(std::abs(*estimate - 0.011) < 1e-9);
    assert(profiler.sample_count(
        TaskType::VectorAdd, 65536, target) == 3);

    // Failed executions must not pollute the profile.
    profiler.record({
        TaskType::VectorAdd, 65536, target, 100.0, false
    });

    assert(profiler.sample_count(
        TaskType::VectorAdd, 65536, target) == 3);

    std::cout << "Profiler tests passed.\n";
    return 0;
}
