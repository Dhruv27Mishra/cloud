#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
from typing import Dict, List


def iter_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        # Some Alibaba files have no header; we inspect first row length.
        first = f.readline().strip()
        if not first:
            return
        parts = first.split(",")
        # Header should contain canonical field names, not task identifiers like M1/R2_1.
        lowered = [p.strip().lower() for p in parts]
        header_markers = {"task_name", "instance_num", "job_name", "start_time", "end_time"}
        has_header = any(x in header_markers for x in lowered)
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
    ap.add_argument("--batch_task_csv", type=str, required=True)
    ap.add_argument("--out_jobs_csv", type=str, required=True)
    ap.add_argument("--max_rows", type=int, default=20000)
    args = ap.parse_args()

    src = Path(args.batch_task_csv)
    dst = Path(args.out_jobs_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, float]] = []
    jid = 0
    t0 = None
    for row in iter_rows(src):
        if isinstance(row, dict):
            st = row.get("start_time") or row.get("starttime") or row.get("start")
            et = row.get("end_time") or row.get("endtime") or row.get("end")
            cpu = row.get("cpu") or row.get("cpu_request") or row.get("plan_cpu")
            mem = row.get("mem") or row.get("memory") or row.get("plan_mem")
            ins = row.get("instance_num") or row.get("inst_num") or "1"
        else:
            # common positional fallback for batch_task style rows:
            # task_name,instance_num,job_name,task_type,status,start_time,end_time,plan_cpu,plan_mem
            if len(row) < 9:
                continue
            st = row[5]
            et = row[6]
            cpu = row[7]
            mem = row[8]
            ins = row[1]
        try:
            stf = float(st)
            etf = float(et) if et not in (None, "", "null", "NULL") else stf + 60.0
        except Exception:
            continue
        if t0 is None:
            t0 = stf
        arrival = max(0.0, stf - t0)
        runtime = max(1.0, etf - stf)
        try:
            cpu_d = max(0.05, min(1.0, float(cpu) / 100.0 if float(cpu) > 1 else float(cpu)))
        except Exception:
            cpu_d = 0.3
        try:
            mem_d = max(0.05, min(1.0, float(mem) / 100.0 if float(mem) > 1 else float(mem)))
        except Exception:
            mem_d = 0.25
        try:
            inst = float(ins)
        except Exception:
            inst = 1.0
        pr = 2 if inst >= 10 else (1 if inst >= 3 else 0)
        ddl = arrival + runtime * (2.1 - 0.2 * pr)
        jobs.append(
            {
                "job_id": jid,
                "arrival_time": arrival,
                "runtime": runtime,
                "cpu_demand": cpu_d,
                "mem_demand": mem_d,
                "priority": pr,
                "deadline_time": ddl,
                "scenario_id": "alibaba_batch_real_slice",
            }
        )
        jid += 1
        if jid >= args.max_rows:
            break

    if not jobs:
        raise SystemExit("No jobs parsed from Alibaba batch_task")
    fields = ["job_id", "arrival_time", "runtime", "cpu_demand", "mem_demand", "priority", "deadline_time", "scenario_id"]
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(jobs)
    print(f"Wrote {dst} with {len(jobs)} rows")


if __name__ == "__main__":
    main()
