cmake_minimum_required(VERSION 3.16)

project(HeteroOS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# --------------------------------------------------
# NVML
# --------------------------------------------------

find_path(NVML_INCLUDE_DIR
    NAMES nvml.h
    HINTS
        /usr/local/cuda/include
        /usr/local/cuda-13.2/targets/x86_64-linux/include
)

find_library(NVML_LIBRARY
    NAMES nvidia-ml libnvidia-ml.so.1
    PATHS
        /lib64
        /usr/lib64
        /usr/local/cuda/lib64
)

if(NOT NVML_INCLUDE_DIR OR NOT NVML_LIBRARY)
    message(FATAL_ERROR "NVML headers or library not found")
endif()


# --------------------------------------------------
# HeteroOS Monitor
# --------------------------------------------------

add_executable(heteroos-monitor
    app/main.cpp

    src/monitoring/cpu_monitor.cpp
    src/monitoring/memory_monitor.cpp
    src/monitoring/gpu_monitor.cpp
    src/monitoring/resource_monitor.cpp

    src/ui/terminal_dashboard.cpp
)

target_include_directories(heteroos-monitor PRIVATE
    include
    ${NVML_INCLUDE_DIR}
)

target_link_libraries(heteroos-monitor PRIVATE
    ${NVML_LIBRARY}
)

target_compile_options(heteroos-monitor PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)


# --------------------------------------------------
# CPU Benchmark
# --------------------------------------------------

add_executable(cpu-benchmark
    app/cpu_benchmark.cpp
    src/execution/cpu_executor.cpp
)

target_include_directories(cpu-benchmark PRIVATE
    include
)

target_compile_options(cpu-benchmark PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)


# --------------------------------------------------
# GPU Benchmark
# --------------------------------------------------

add_executable(gpu-benchmark
    app/gpu_benchmark.cpp
    src/execution/gpu_executor.cu
)

target_include_directories(gpu-benchmark PRIVATE
    include
)

target_link_libraries(gpu-benchmark PRIVATE
    CUDA::cudart
)

target_compile_options(gpu-benchmark PRIVATE
    -Wall
    -Wextra
)


# --------------------------------------------------
# Unified Execution Benchmark
# --------------------------------------------------

add_executable(execution-benchmark
    app/execution_benchmark.cpp
    src/execution/cpu_executor.cpp
    src/execution/gpu_executor.cu
)

target_include_directories(execution-benchmark PRIVATE
    include
)

target_link_libraries(execution-benchmark PRIVATE
    CUDA::cudart
)

target_compile_options(execution-benchmark PRIVATE
    -Wall
    -Wextra
)








add_executable(cpu-gpu-scaling-benchmark
    app/cpu_gpu_scaling_benchmark.cpp
    src/execution/cpu_executor.cpp
)

target_include_directories(cpu-gpu-scaling-benchmark PRIVATE
    include
)

target_compile_options(cpu-gpu-scaling-benchmark PRIVATE
    -Wall
    -Wextra
)




add_executable(vector-add-benchmark
    app/vector_add_benchmark.cpp
    src/execution/cpu_executor.cpp
    src/execution/gpu_executor.cu
)

target_include_directories(vector-add-benchmark PRIVATE
    include
)

target_link_libraries(vector-add-benchmark PRIVATE
    CUDA::cudart
)

target_compile_options(vector-add-benchmark PRIVATE
    -Wall
    -Wextra
)





# --------------------------------------------------
# Task Lifecycle Demo
# --------------------------------------------------

add_executable(task-demo
    app/task_demo.cpp
    src/task/task.cpp
)

target_include_directories(task-demo PRIVATE
    include
)

target_compile_options(task-demo PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)


# --------------------------------------------------
# Task Manager Demo
# --------------------------------------------------

add_executable(task-manager-demo
    app/task_manager_demo.cpp
    src/task/task.cpp
    src/task/task_manager.cpp
)

target_include_directories(task-manager-demo PRIVATE
    include
)

target_compile_options(task-manager-demo PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)



# --------------------------------------------------
# Execution Coordinator Demo
# --------------------------------------------------

add_executable(execution-coordinator-demo
    app/execution_coordinator_demo.cpp

    src/task/task.cpp
    src/task/task_manager.cpp

    src/scheduling/scheduler.cpp

    src/execution/execution_coordinator.cpp
    src/execution/cpu_executor.cpp
    src/execution/gpu_executor.cu

    src/monitoring/cpu_monitor.cpp
    src/monitoring/memory_monitor.cpp
    src/monitoring/gpu_monitor.cpp
    src/monitoring/resource_monitor.cpp

    src/profiling/profiler.cpp
)

target_include_directories(execution-coordinator-demo PRIVATE
    include
    ${NVML_INCLUDE_DIR}
)

target_link_libraries(execution-coordinator-demo PRIVATE
    CUDA::cudart
    ${NVML_LIBRARY}
)

target_compile_options(execution-coordinator-demo PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>
)


add_executable(paired-execution-benchmark
    app/paired_execution_benchmark.cpp
    src/execution/cpu_executor.cpp
    src/execution/gpu_executor.cu
)

target_include_directories(paired-execution-benchmark PRIVATE
    include
)

target_link_libraries(paired-execution-benchmark PRIVATE
    CUDA::cudart
)

target_compile_options(paired-execution-benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>
)



add_executable(gpu-timing-diagnostic
    app/gpu_timing_diagnostic.cpp
    src/execution/gpu_executor.cu
)

target_include_directories(gpu-timing-diagnostic PRIVATE
    include
)

target_link_libraries(gpu-timing-diagnostic PRIVATE
    CUDA::cudart
)

target_compile_options(gpu-timing-diagnostic PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>
)



add_executable(coordinator-timing-diagnostic
    app/coordinator_timing_diagnostic.cpp
    src/task/task.cpp
    src/task/task_manager.cpp
    src/execution/execution_coordinator.cpp
    src/execution/cpu_executor.cpp
    src/execution/gpu_executor.cu
)

target_include_directories(coordinator-timing-diagnostic PRIVATE
    include
)

target_link_libraries(coordinator-timing-diagnostic PRIVATE
    CUDA::cudart
)

target_compile_options(coordinator-timing-diagnostic PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>
)



add_executable(gpu-cold-start-diagnostic
    app/gpu_cold_start_diagnostic.cpp
    src/execution/gpu_executor.cu
)

target_include_directories(gpu-cold-start-diagnostic PRIVATE
    include
)

target_link_libraries(gpu-cold-start-diagnostic PRIVATE
    CUDA::cudart
)

target_compile_options(gpu-cold-start-diagnostic PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>
)




add_executable(profiler-test
    app/profiler_test.cpp
    src/profiling/profiler.cpp
    src/task/task.cpp
)

target_include_directories(profiler-test PRIVATE include)

target_compile_options(profiler-test PRIVATE
    -Wall -Wextra -Wpedantic
)




add_executable(profiler-calibration
    app/profiler_calibration.cpp

    src/task/task.cpp
    src/task/task_manager.cpp

    src/execution/execution_coordinator.cpp
    src/execution/cpu_executor.cpp
    src/execution/gpu_executor.cu

    src/profiling/profiler.cpp
)

target_include_directories(profiler-calibration PRIVATE include)

target_compile_options(profiler-calibration PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)