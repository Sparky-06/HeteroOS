
#include <linux/atomic.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("HeteroOS");
MODULE_DESCRIPTION("HeteroOS kernel event monitor");
MODULE_VERSION("0.2");

#define PROC_DIR_NAME   "heteroos"
#define PROC_FILE_NAME  "events"

static struct proc_dir_entry *heteroos_dir;
static struct proc_dir_entry *events_file;

static atomic64_t cpu_tasks = ATOMIC64_INIT(0);
static atomic64_t gpu_tasks = ATOMIC64_INIT(0);

static ssize_t heteroos_read(
    struct file *file,
    char __user *user_buffer,
    size_t count,
    loff_t *position)
{
    char buffer[128];
    int length;

    length = scnprintf(
        buffer,
        sizeof(buffer),
        "cpu_tasks: %lld\ngpu_tasks: %lld\ntotal_tasks: %lld\n",
        (long long)atomic64_read(&cpu_tasks),
        (long long)atomic64_read(&gpu_tasks),
        (long long)(atomic64_read(&cpu_tasks) +
                    atomic64_read(&gpu_tasks))
    );

    return simple_read_from_buffer(
        user_buffer, count, position, buffer, length);
}

static ssize_t heteroos_write(
    struct file *file,
    const char __user *user_buffer,
    size_t count,
    loff_t *position)
{
    char event[16];
    char *name;

    if (count == 0 || count >= sizeof(event))
        return -EINVAL;

    if (copy_from_user(event, user_buffer, count))
        return -EFAULT;

    event[count] = '\0';
    name = strim(event);

    if (strcmp(name, "CPU") == 0) {
        atomic64_inc(&cpu_tasks);
    } else if (strcmp(name, "GPU") == 0) {
        atomic64_inc(&gpu_tasks);
    } else {
        pr_warn("heteroos: rejected unknown event\n");
        return -EINVAL;
    }

    pr_debug("heteroos: recorded %s task completion\n", name);
    return count;
}

static const struct proc_ops heteroos_proc_ops = {
    .proc_read  = heteroos_read,
    .proc_write = heteroos_write,
};

static int __init heteroos_monitor_init(void)
{
    heteroos_dir = proc_mkdir(PROC_DIR_NAME, NULL);
    if (!heteroos_dir) {
        pr_err("heteroos: failed to create /proc/%s\n",
               PROC_DIR_NAME);
        return -ENOMEM;
    }

    /*
     * Demo permissions: any local user can report events.
     * Production deployments should restrict write access.
     */
    events_file = proc_create(
        PROC_FILE_NAME, 0666, heteroos_dir, &heteroos_proc_ops);

    if (!events_file) {
        proc_remove(heteroos_dir);
        heteroos_dir = NULL;
        pr_err("heteroos: failed to create events interface\n");
        return -ENOMEM;
    }

    pr_info("heteroos: kernel event monitor loaded\n");
    pr_info("heteroos: interface available at /proc/%s/%s\n",
            PROC_DIR_NAME, PROC_FILE_NAME);

    return 0;
}

static void __exit heteroos_monitor_exit(void)
{
    if (events_file)
        proc_remove(events_file);

    if (heteroos_dir)
        proc_remove(heteroos_dir);

    pr_info("heteroos: kernel event monitor unloaded\n");
}

module_init(heteroos_monitor_init);
module_exit(heteroos_monitor_exit);
