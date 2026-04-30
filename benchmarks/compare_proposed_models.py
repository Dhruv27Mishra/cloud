#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from benchmarks.data.job_dataset import load_jobs_csv
from benchmarks.envs.trace_scheduling_env import TraceSchedulingEnv
from benchmarks.models.proposed_models import train_hybrid_role, train_performative_rl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="benchmarks/outputs/proposed_compare")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_queue", type=int, default=16)
    return ap.parse_args()


def make_env(path: str, max_queue: int, max_steps: int) -> TraceSchedulingEnv:
    jobs = load_jobs_csv(path)
    return TraceSchedulingEnv(jobs=jobs, max_queue=max_queue, max_steps=max_steps)


def _score(row: Dict[str, str]) -> float:
    # higher is better: return - miss penalty - wait penalty - energy penalty
    r = float(row["mean_return"])
    m = float(row["mean_deadline_misses"])
    w = float(row["mean_wait"])
    e = float(row["mean_energy"])
    return r - 0.25 * m - 0.05 * w - 0.05 * e


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trainers: List[Tuple[str, callable]] = [
        ("performative_rl", train_performative_rl),
        ("hybrid_role_based_marl", train_hybrid_role),
    ]

    rows: List[Dict[str, str]] = []
    for name, trainer in trainers:
        print(f"Training {name} ...", flush=True)
        env = make_env(args.jobs_csv, args.max_queue, args.steps)
        res = trainer(env, args.steps, args.seed, out / "models")
        row = {
            "model": name,
            "model_path": res.model_path,
            "mean_return": f"{res.mean_return:.6f}",
            "mean_deadline_misses": f"{res.mean_deadline_misses:.6f}",
            "mean_wait": f"{res.mean_wait:.6f}",
            "mean_energy": f"{res.mean_energy:.6f}",
        }
        row["composite_score"] = f"{_score(row):.6f}"
        rows.append(row)

    rows.sort(key=lambda x: float(x["composite_score"]), reverse=True)
    if rows:
        rows[0]["is_best"] = "yes"
        for r in rows[1:]:
            r["is_best"] = "no"
    fields = [
        "model",
        "composite_score",
        "mean_return",
        "mean_deadline_misses",
        "mean_wait",
        "mean_energy",
        "is_best",
        "model_path",
    ]
    with (out / "proposed_model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if rows:
        best = rows[0]
        (out / "BEST_MODEL.txt").write_text(
            "\n".join(
                [
                    f"Best model: {best['model']}",
                    f"Composite score: {best['composite_score']}",
                    f"Mean return: {best['mean_return']}",
                    f"Deadline misses: {best['mean_deadline_misses']}",
                    f"Mean wait: {best['mean_wait']}",
                    f"Mean energy: {best['mean_energy']}",
                    f"Checkpoint: {best['model_path']}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {out / 'proposed_model_comparison.csv'}")


if __name__ == "__main__":
    main()
