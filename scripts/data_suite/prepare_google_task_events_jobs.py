#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
from typing import Dict, List, Tuple


"""
Google task_events schema (v2.1) fields:
0 time
1 missing info
2 job ID
3 task index
4 machine ID
5 event type
6 user
7 scheduling class
8 priority
9 cpu request
10 memory request
11 disk request
12 different-machine constraint

We create one "job-like" row per (job_id, task_index) using first submit event and first terminal event.
"""


SUBMIT_EVENT = 0
TERMINAL_EVENTS = {2, 3, 4, 5, 6, 8}  # fail/evict/finish/kill/lost etc.


def iter_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            yield row


def build_jobs(path: Path, max_tasks: int) -> List[Dict[str, float]]:
    starts: Dict[Tuple[str, str], Dict[str, float]] = {}
    ends: Dict[Tuple[str, str], float] = {}

    for row in iter_rows(path):
        if len(row) < 11:
            continue
        t = float(row[0])
        job = row[2]
        task = row[3]
        key = (job, task)
        evt = int(float(row[5]))
        if evt == SUBMIT_EVENT and key not in starts:
            pr = float(row[8]) if row[8] else 0.0
            cpu = float(row[9]) if row[9] else 0.1
            mem = float(row[10]) if row[10] else 0.1
            starts[key] = {"arrival_time": t, "priority_raw": pr, "cpu": max(cpu, 0.01), "mem": max(mem, 0.01)}
        elif evt in TERMINAL_EVENTS and key in starts and key not in ends:
            ends[key] = t
        if len(starts) >= max_tasks * 3 and len(ends) >= max_tasks:
            # enough headroom; stop early for quick experiments
            break

    # Normalize priorities into {0,1,2}
    pr_vals = [v["priority_raw"] for v in starts.values()]
    if pr_vals:
        p_lo = sorted(pr_vals)[int(0.33 * (len(pr_vals) - 1))]
        p_hi = sorted(pr_vals)[int(0.66 * (len(pr_vals) - 1))]
    else:
        p_lo, p_hi = 1.0, 2.0

    jobs: List[Dict[str, float]] = []
    jid = 0
    for k, s in starts.items():
        if k not in ends:
            continue
        arrival = s["arrival_time"]
        end = ends[k]
        runtime = max(1.0, end - arrival)
        pr_raw = s["priority_raw"]
        if pr_raw <= p_lo:
            pri = 0
        elif pr_raw <= p_hi:
            pri = 1
        else:
            pri = 2
        deadline = arrival + runtime * (2.0 - 0.25 * pri)
        jobs.append(
            {
                "job_id": jid,
                "arrival_time": arrival,
                "runtime": runtime,
                "cpu_demand": min(1.0, s["cpu"]),
                "mem_demand": min(1.0, s["mem"]),
                "priority": pri,
                "deadline_time": deadline,
                "scenario_id": "google_cluster_real_slice",
            }
        )
        jid += 1
        if jid >= max_tasks:
            break
    jobs.sort(key=lambda x: x["arrival_time"])
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_events_csv", type=str, required=True)
    ap.add_argument("--out_jobs_csv", type=str, required=True)
    ap.add_argument("--max_tasks", type=int, default=20000)
    args = ap.parse_args()

    src = Path(args.task_events_csv)
    dst = Path(args.out_jobs_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(src, args.max_tasks)
    if not jobs:
        raise SystemExit("No jobs parsed from task_events")
    fields = ["job_id", "arrival_time", "runtime", "cpu_demand", "mem_demand", "priority", "deadline_time", "scenario_id"]
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(jobs)
    print(f"Wrote {dst} with {len(jobs)} rows")


if __name__ == "__main__":
    main()
