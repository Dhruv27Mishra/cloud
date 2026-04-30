#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from benchmarks.compare_full_suite import _make_env
from benchmarks.data.job_dataset import load_jobs_csv
from benchmarks.plot_convergence_noisy import _build_act_fn


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite_root",
        type=str,
        default="datasets/synthetic/suite_v3_large",
        help="Synthetic suite directory containing scenario_xxx/jobs.csv.",
    )
    ap.add_argument(
        "--compare_root",
        type=str,
        default="benchmarks/outputs/full_synthetic_v3_whole_suite_pub",
        help="Per-scenario trained model comparison outputs.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="benchmarks/outputs/convergence_synthetic_v3_a2c_units/convergence_stepwise_all_scenarios.csv",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument(
        "--exclude_algos",
        type=str,
        nargs="+",
        default=["cql_discrete"],
        help="Algorithms to exclude from convergence export.",
    )
    ap.add_argument("--max_queue", type=int, default=16)
    ap.add_argument("--cluster_cores", type=float, default=8.0)
    ap.add_argument("--cluster_mem", type=float, default=32.0)
    return ap.parse_args()


def _queue_load_metrics(env) -> Dict[str, float]:
    q = list(getattr(env, "queue", []))
    if not q:
        return {
            "cpu_util_pct": 0.0,
            "mem_util_pct": 0.0,
            "io_util_pct": 0.0,
            "load_balance_factor": 0.0,
        }

    cpu = np.asarray([float(j.cpu_demand) for j in q], dtype=np.float64)
    mem = np.asarray([float(j.mem_demand) for j in q], dtype=np.float64)
    run = np.asarray([float(j.runtime) for j in q], dtype=np.float64)

    cpu_util = 100.0 * float(np.sum(cpu) / max(float(env.cluster_cores), 1e-6))
    mem_util = 100.0 * float(np.sum(mem) / max(float(env.cluster_mem), 1e-6))
    # A2C paper uses CPU/I/O/MEM. Environment has no explicit I/O; use normalized runtime pressure as I/O proxy.
    io_proxy = 100.0 * float(np.mean(np.clip(run / (np.mean(run) + 1e-6), 0.0, 3.0)) / 3.0)

    # A2C load-balancing factor uses dispersion; use std of per-job normalized demand.
    demand = 0.5 * cpu + 0.5 * mem
    lbf = float(np.std(demand)) if len(demand) > 1 else 0.0
    return {
        "cpu_util_pct": cpu_util,
        "mem_util_pct": mem_util,
        "io_util_pct": io_proxy,
        "load_balance_factor": lbf,
    }


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root)
    compare_root = Path(args.compare_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    scenario_dirs = sorted([p for p in suite_root.glob("scenario_*") if p.is_dir()])
    all_rows: List[Dict[str, str]] = []

    for sdir in scenario_dirs:
        sid = sdir.name
        jobs_csv = sdir / "jobs.csv"
        comp_csv = compare_root / sid / "full_suite_comparison.csv"
        if not jobs_csv.exists() or not comp_csv.exists():
            continue

        with comp_csv.open(newline="", encoding="utf-8") as f:
            models = list(csv.DictReader(f))
        n_jobs = len(load_jobs_csv(jobs_csv))

        for m in models:
            algo = m["algo"]
            if algo in set(args.exclude_algos):
                continue
            model_path = m["model_path"]
            for seed in args.seeds:
                env = _make_env(
                    str(jobs_csv),
                    max_queue=args.max_queue,
                    cluster_cores=args.cluster_cores,
                    cluster_mem=args.cluster_mem,
                    max_steps=args.max_steps,
                )
                obs, _ = env.reset(seed=seed)
                act_fn = _build_act_fn(algo, model_path, env)
                done = trunc = False
                step = 0
                while not done and not trunc and step < args.max_steps:
                    step += 1
                    a = act_fn(obs)
                    obs, r, done, trunc, info = env.step(a)
                    q_metrics = _queue_load_metrics(env)
                    arrived = max(int(getattr(env, "idx", 0)), 1)
                    rej_rate = 100.0 * float(info.get("dismissals_total", 0.0)) / float(arrived)

                    all_rows.append(
                        {
                            "dataset": sid,
                            "algo": algo,
                            "seed": str(seed),
                            "episode": str(step),  # A2C-style naming
                            "timestamp_min": str(step),  # 1-min synthetic time resolution
                            "reward": f"{float(r):.8f}",
                            "throughput_tasks_per_min": f"{float(info.get('completed', 0.0)) / max(float(info.get('time', 1.0)), 1.0):.8f}",
                            "task_rejection_rate_pct": f"{rej_rate:.8f}",
                            "deadline_miss_rate_pct": f"{100.0 * float(info.get('deadline_misses', 0.0)) / max(float(info.get('completed', 1.0)), 1.0):.8f}",
                            "mean_wait_time": f"{float(info.get('mean_wait', 0.0)):.8f}",
                            "mean_energy_per_task": f"{float(info.get('mean_energy', 0.0)):.8f}",
                            "load_balance_factor": f"{q_metrics['load_balance_factor']:.8f}",
                            "cpu_util_pct": f"{q_metrics['cpu_util_pct']:.8f}",
                            "io_util_pct": f"{q_metrics['io_util_pct']:.8f}",
                            "mem_util_pct": f"{q_metrics['mem_util_pct']:.8f}",
                            "fairness_variance_cpu": f"{float(env._fairness_variance_cpu()):.8f}",
                        }
                    )

    fields = [
        "dataset",
        "algo",
        "seed",
        "episode",
        "timestamp_min",
        "reward",
        "throughput_tasks_per_min",
        "task_rejection_rate_pct",
        "deadline_miss_rate_pct",
        "mean_wait_time",
        "mean_energy_per_task",
        "load_balance_factor",
        "cpu_util_pct",
        "io_util_pct",
        "mem_util_pct",
        "fairness_variance_cpu",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {out_csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()

