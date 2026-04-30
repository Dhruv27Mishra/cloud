#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
from typing import Dict, List


def read_csv_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        first = f.readline().strip()
        if not first:
            return
        cols = first.split(",")
        # Azure vmtable in this release is typically headerless with 11 columns.
        header_tokens = {"vmid", "avgcpu", "vmcreated", "vmdeleted", "subscriptionid"}
        has_header = any(c.strip().lower() in header_tokens for c in cols)
        f.seek(0)
        if has_header:
            r = csv.DictReader(f)
            for row in r:
                yield row
        else:
            r = csv.reader(f)
            for row in r:
                yield row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmtable_csv", type=str, required=True)
    ap.add_argument("--out_jobs_csv", type=str, required=True)
    ap.add_argument("--max_rows", type=int, default=20000)
    args = ap.parse_args()

    src = Path(args.vmtable_csv)
    dst = Path(args.out_jobs_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, float]] = []
    t0 = None
    jid = 0
    for row in read_csv_rows(src):
        # Supports both headered dict rows and headerless positional rows.
        if isinstance(row, dict):
            ts_create = row.get("vmcreated") or row.get("vmCreated") or row.get("timestamp_vm_created") or row.get("3")
            ts_del = row.get("vmdeleted") or row.get("vmDeleted") or row.get("timestamp_vm_deleted") or row.get("4")
            avgcpu = row.get("avgcpu") or row.get("avgCpu") or row.get("5")
            cores = row.get("vmcorecountbucket") or row.get("cores") or row.get("9")
            mem = row.get("vmmemorybucket") or row.get("memory") or row.get("10")
        else:
            if len(row) < 11:
                continue
            # Observed schema (headerless):
            # 0 sub_id,1 dep_id,2 vm_id,3 vm_created,4 vm_deleted,5 maxcpu,6 avgcpu,7 p95maxcpu,8 category,9 cores,10 memory
            ts_create = row[3]
            ts_del = row[4]
            avgcpu = row[6]
            cores = row[9]
            mem = row[10]
        if ts_create is None or ts_del is None:
            continue
        try:
            t_c = float(ts_create)
            t_d = float(ts_del)
        except Exception:
            continue
        if t0 is None:
            t0 = t_c
        arrival = max(0.0, t_c - t0)
        runtime = max(1.0, t_d - t_c if t_d > t_c else 300.0)
        try:
            cpu = float(avgcpu) / 100.0 if avgcpu is not None else 0.3
        except Exception:
            cpu = 0.3
        cpu = max(0.05, min(1.0, cpu))
        # coarse memory normalization from bucket id/value
        try:
            mem_raw = float(mem) if mem is not None else 8.0
        except Exception:
            mem_raw = 8.0
        mem_d = max(0.05, min(1.0, mem_raw / 64.0))
        try:
            c_raw = float(cores) if cores is not None else 2.0
        except Exception:
            c_raw = 2.0
        pr = 2 if (cpu > 0.75 or c_raw >= 8.0) else (1 if cpu > 0.35 else 0)
        ddl = arrival + runtime * (2.3 - 0.25 * pr)
        jobs.append(
            {
                "job_id": jid,
                "arrival_time": arrival,
                "runtime": runtime,
                "cpu_demand": cpu,
                "mem_demand": mem_d,
                "priority": pr,
                "deadline_time": ddl,
                "scenario_id": "azure_vm_real_slice",
            }
        )
        jid += 1
        if jid >= args.max_rows:
            break

    if not jobs:
        raise SystemExit("No jobs parsed from Azure vmtable")
    fields = ["job_id", "arrival_time", "runtime", "cpu_demand", "mem_demand", "priority", "deadline_time", "scenario_id"]
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(jobs)
    print(f"Wrote {dst} with {len(jobs)} rows")


if __name__ == "__main__":
    main()
