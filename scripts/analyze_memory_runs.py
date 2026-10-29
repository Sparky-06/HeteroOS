
#!/usr/bin/env python3

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

if len(sys.argv) < 2:
    print(
        "Usage: python3 scripts/analyze_memory_runs.py "
        "run1.csv run2.csv ..."
    )
    raise SystemExit(1)

measurements = defaultdict(list)

for filename in sys.argv[1:]:
    path = Path(filename)

    with path.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["correct"].lower() != "true":
                raise SystemExit(f"Incorrect result in {path}")

            # Support both benchmark CSV schemas.
            size = int(
                row["total_bytes"]
                if "total_bytes" in row
                else int(row["buffer_bytes"]) * 2
            )

            bandwidth = float(row["effective_copy_gbps"])
            measurements[size].append(bandwidth)

print(
    f"{'Working MiB':>12} {'Runs':>6} {'Mean GB/s':>12} "
    f"{'Std dev':>12} {'CV %':>10}"
)

for size in sorted(measurements):
    values = measurements[size]
    mean = statistics.mean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = 100.0 * stddev / mean if mean else 0.0

    print(
        f"{size / 1048576:12.2f} {len(values):6d} "
        f"{mean:12.2f} {stddev:12.2f} {cv:10.2f}"
    )
