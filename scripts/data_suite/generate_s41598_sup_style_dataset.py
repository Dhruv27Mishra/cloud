#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_len_rng(s: str) -> Tuple[int, int]:
    lo, hi = s.split("-")
    return int(lo), int(hi)


def _load_specs(task_json: Path, vm_json: Path) -> Tuple[List[Dict], List[Dict]]:
    with task_json.open(encoding="utf-8") as f:
        t = json.load(f)
    with vm_json.open(encoding="utf-8") as f:
        v = json.load(f)
    return t, v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_json", type=str, default="/tmp/rlmots_sup/data/task_classes.json")
    ap.add_argument("--vm_json", type=str, default="/tmp/rlmots_sup/data/vm_types.json")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--n_jobs", type=int, default=120000)
    ap.add_argument("--horizon_minutes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    task_specs, vm_specs = _load_specs(Path(args.task_json), Path(args.vm_json))
    mips_vals = [float(v["MIPS"]) for v in vm_specs]
    max_mips = max(mips_vals)

    rows = []
    arr_t = 0
    for jid in range(args.n_jobs):
        # Non-stationary arrivals with burst probability.
        base_gap = 1 if rng.random() < 0.7 else 2
        if rng.random() < 0.02:
            base_gap = 0
        arr_t = min(args.horizon_minutes, arr_t + base_gap)

        cls = task_specs[rng.randrange(len(task_specs))]
        lo, hi = _parse_len_rng(cls["Length"])
        length = rng.randint(lo, hi)  # MI
        # Convert to normalized runtime-like signal for our env.
        vm = vm_specs[rng.randrange(len(vm_specs))]
        runtime = max(1.0, length / max(float(vm["MIPS"]), 1e-6))
        # CPU/MEM demand normalized to cluster-style range.
        cpu = min(1.0, max(0.05, float(vm["MIPS"]) / max_mips + rng.uniform(-0.05, 0.05)))
        mem = min(1.0, max(0.05, float(vm["RAM"]) / max(float(v["RAM"]) for v in vm_specs) + rng.uniform(-0.08, 0.08)))
        # Priority tied to task class: I lowest, III highest.
        prio = {"I": 0, "II": 1, "III": 2}.get(cls["Class"], 1)
        # Tight/medium/loose deadlines mixed for complexity.
        ddl_mult = {0: rng.uniform(2.0, 4.0), 1: rng.uniform(1.5, 3.0), 2: rng.uniform(1.1, 2.2)}[prio]
        deadline = arr_t + runtime * ddl_mult

        rows.append(
            {
                "job_id": jid,
                "arrival_time": float(arr_t),
                "runtime": float(runtime),
                "cpu_demand": float(cpu),
                "mem_demand": float(mem),
                "priority": int(prio),
                "deadline_time": float(deadline),
            }
        )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["job_id", "arrival_time", "runtime", "cpu_demand", "mem_demand", "priority", "deadline_time"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} jobs)")


if __name__ == "__main__":
    main()

