# HeteroOS — Version 1

**An experimental Linux userspace runtime for heterogeneous CPU–GPU computing**

HeteroOS monitors system resources, estimates execution costs for supported computational workloads, and selects a CPU or GPU execution path. It combines a userspace runtime with a small Linux kernel module for execution-event telemetry.

> **Project scope:** HeteroOS is an experimental runtime for workloads it explicitly supports. It does not replace the Linux kernel scheduler or transparently move arbitrary Linux processes between CPU and GPU.

## Overview

CPUs and GPUs have different performance characteristics. A GPU can offer substantial acceleration for some parallel workloads, but launch overhead, data movement, workload size, and available resources can make CPU execution the better choice for others.

HeteroOS explores this decision by combining:

- System resource monitoring
- CPU and CUDA execution paths
- Workload cost estimation and CPU/GPU selection
- Execution profiling and benchmark measurements
- Kernel-level telemetry for HeteroOS-reported execution events

Version 1 focuses on a working end-to-end path: submit a supported task, choose an eligible execution target, run it, validate the result, and record timing and execution information.

## Features

- **Resource monitoring:** Collects CPU, memory, and NVIDIA GPU information using Linux interfaces and NVML.
- **CPU execution:** Runs supported workloads through the CPU execution engine, including serial and parallel implementations where available.
- **GPU execution:** Runs supported workloads through CUDA.
- **Cost-based scheduling:** Estimates CPU and GPU execution costs and selects an eligible target.
- **Execution coordination:** Connects task submission, resource collection, scheduling, execution, validation, and profiling.
- **Profiling and benchmarks:** Records timing measurements and exports results to CSV.
- **Kernel event telemetry:** A Linux kernel module maintains CPU/GPU event counters exposed through `/proc/heteroos/events`.

## Architecture

```text
                    HeteroOS userspace runtime
┌───────────────────────────────────────────────────────────┐
│ Task submission and task management                       │
│                    │                                      │
│                    ▼                                      │
│ Resource monitoring ──► Cost estimation ──► Scheduler     │
│                                              │            │
│                              ┌───────────────┴──────────┐ │
│                              ▼                          ▼ │
│                        CPU executor               CUDA executor
│                              └───────────────┬──────────┘ │
│                                              ▼            │
│                                 Result validation         │
│                                              │            │
│                                 Profiling / CSV output    │
│                                              │            │
│                                 Kernel event client       │
└──────────────────────────────────────────────┼────────────┘
                                               ▼
                                   Linux kernel module
                                               │
                                               ▼
                                  /proc/heteroos/events
```

The scheduler runs in userspace. The kernel module records events reported by the runtime; it does not make scheduling decisions.

## Technology Stack

- C and C++
- CUDA C++
- Linux `/proc` and `/sys`
- NVIDIA Management Library (NVML)
- Linux kernel modules
- CMake
- Fedora Linux

## Supported Workloads

Version 1 includes CPU and GPU implementations for selected computational workloads, including:

- Vector addition
- Matrix multiplication

The exact workload types and supported sizes depend on the benchmark or demo executable being used. HeteroOS does not currently execute arbitrary applications on the GPU.

## Requirements

- Linux system with a C++ compiler and CMake
- NVIDIA GPU and a working NVIDIA driver for CUDA execution
- CUDA Toolkit
- NVML development headers and library
- Linux kernel development files matching the running kernel, if building the kernel module
- Root privileges to load or unload the kernel module

The CPU-side components and GPU/kernel components have different requirements. A machine without a supported NVIDIA GPU may still be useful for examining or developing CPU-side components, but CUDA targets require a working CUDA environment.

## Build

From the project root:

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

CMake configuration requires the CUDA Toolkit and NVML dependencies specified by the project.

## Run

### Resource monitor

```bash
./build/heteroos-monitor
```

### Execution coordinator demo

```bash
./build/execution-coordinator-demo
```

The coordinator demo runs the configured tasks, displays scheduling and execution information, validates results, and attempts to report execution events to the kernel module when it is available.

### Benchmarks

Build the project first, then run the benchmark executables from `build/`. For example:

```bash
./build/cpu-benchmark
./build/gpu-benchmark
./build/paired-execution-benchmark
./build/memory-bandwidth-benchmark
./build/memory-bandwidth-stream-benchmark
```

Benchmark executables may write CSV output under `build/`. These are generated experiment results, not source files.

## Linux Kernel Module

The HeteroOS kernel module, `heteroos_monitor`, records CPU/GPU execution events reported by the userspace runtime. It exposes cumulative counters through:

```text
/proc/heteroos/events
```

Example output:

```text
cpu_tasks: 1
gpu_tasks: 1
total_tasks: 2
```

The counters are in-memory values maintained by the loaded module. They reset when the module is unloaded and reloaded.

### Build the module

Build against the currently running kernel:

```bash
cd kernel
make clean
make
```

Check that the module's version metadata matches the running kernel:

```bash
uname -r
modinfo heteroos_monitor.ko | grep vermagic
```

### Secure Boot

If Secure Boot is enabled, sign the rebuilt module using an enrolled signing key before loading it. The signing key should be kept outside the repository and must not be committed.

### Load, inspect, and unload

```bash
sudo insmod heteroos_monitor.ko
cat /proc/heteroos/events
```

From the project root, run the coordinator demo and inspect the counters again:

```bash
./build/execution-coordinator-demo
cat /proc/heteroos/events
```

Unload the module when finished:

```bash
sudo rmmod heteroos_monitor
```

The procfs interface is intended for project experimentation. Its current permissive write access allows local users to submit synthetic events, so the counters should be treated as telemetry—not as a trusted security or scheduling authority.

## Benchmarking and Evaluation

HeteroOS uses repeated measurements, warm-up runs, median and P95 timings, correctness checks, and CSV output to examine execution performance.

The benchmark results demonstrate that the preferred execution target depends on the workload and its size. In the tested cases, vector addition favored CPU execution, while matrix multiplication favored GPU execution. These observations apply to the tested implementations and conditions; they are not universal claims about CPUs, GPUs, or all workloads.

The memory-copy benchmarks measure effective CPU copy throughput. They do not independently isolate DRAM bandwidth, measure CPU-to-GPU transfer bandwidth, or directly provide a complete scheduler cost estimate.

## Current Limitations

- HeteroOS supports selected workloads rather than arbitrary Linux applications.
- Scheduling estimates are experimental and may differ from measured execution times.
- GPU execution depends on the available hardware, driver, CUDA environment, and device memory.
- The kernel module records events reported by HeteroOS; it does not independently observe all system CPU/GPU work.
- Memory-copy measurements do not represent the full end-to-end cost of a computational task.
- The kernel event interface is designed for experimentation and is not production-hardened.

## Project Status

**Version 1 — Experimental prototype**

The current version demonstrates the core runtime path: resource monitoring, CPU/GPU execution, cost-based target selection, result validation, profiling, and kernel/userspace event reporting.

Future work may include broader workload coverage, improved cost-model calibration, additional scheduling-policy comparisons, and stronger kernel interface controls.

## Repository Layout

```text
HeteroOS/
├── app/          # Demos, benchmarks, and diagnostic programs
├── include/      # Public headers for runtime components
├── src/          # Runtime implementations
├── kernel/       # Kernel module source and Makefile
├── scripts/      # Supporting scripts
├── tests/        # Tests, if present
├── CMakeLists.txt
└── README.md
```
