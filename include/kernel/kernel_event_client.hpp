
#ifndef HETEROOS_KERNEL_EVENT_CLIENT_HPP
#define HETEROOS_KERNEL_EVENT_CLIENT_HPP

#include "task/task.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>

namespace heteroos {


inline bool record_kernel_event(ExecutionTarget target)
{
    const char* event = nullptr;

    switch (target) {
        case ExecutionTarget::CPU:
            event = "CPU\n";
            break;
        case ExecutionTarget::GPU:
            event = "GPU\n";
            break;
        default:
            return false;
    }

    const int fd = open(
        "/proc/heteroos/events",
        O_WRONLY | O_CLOEXEC
    );

    if (fd < 0) {
        std::perror("heteroos event open");
        return false;
    }

    const ssize_t written = write(fd, event, 4);

    if (written < 0) {
        std::perror("heteroos event write");
        close(fd);
        return false;
    }

    close(fd);

    if (written != 4) {
        std::fprintf(
            stderr,
            "heteroos event write: expected 4 bytes, wrote %zd\n",
            written
        );
        return false;
    }

    return true;
}


} // namespace heteroos

#endif
