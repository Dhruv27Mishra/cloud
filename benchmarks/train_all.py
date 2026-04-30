#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

from benchmarks.data.job_dataset import load_jobs_csv
from benchmarks.envs.trace_scheduling_env import RewardWeights, TraceSchedulingEnv
from benchmarks.models.cloud_paper_models import (
    train_decima_style,
    train_deepjs_dqn,
    train_deeprm_pg,
)
from benchmarks.models.paper_models import (
    train_cql_discrete,
    train_decision_transformer,
    train_discrete_sac,
    train_iql_discrete,
)
from benchmarks.models.proposed_models import train_hybrid_role, train_performative_rl
from benchmarks.models.sb3_baselines import train_sb3_baseline


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs_csv", type=str, required=True, help="Path to jobs.csv in unified schema.")
    ap.add_argument("--out_dir", type=str, default="benchmarks/outputs/run_01")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=25_000)
    ap.add_argument(
        "--algos",
        nargs="+",
        default=[
            "ppo",
            "a2c",
            "sac_discrete",
            "dqn",
            "decision_transformer",
            "iql_discrete",
            "cql_discrete",
            "deeprm_pg",
            "decima_style",
            "deepjs_dqn",
            "performative_rl",
            "hybrid_role_based_marl",
        ],
    )
    ap.add_argument("--max_queue", type=int, default=16)
    ap.add_argument("--cluster_cores", type=float, default=8.0)
    ap.add_argument("--cluster_mem", type=float, default=32.0)
    return ap.parse_args()


def make_env(args: argparse.Namespace) -> TraceSchedulingEnv:
    rw = RewardWeights(
        use_wa3c=os.getenv("RW_USE_WA3C", "1") == "1",
        w_qos=float(os.getenv("RW_W_QOS", "0.25")),
        w_energy=float(os.getenv("RW_W_ENERGY", "0.2")),
        w_priority=float(os.getenv("RW_W_PRIORITY", "0.25")),
        w_fair=float(os.getenv("RW_W_FAIR", "0.15")),
        w_dismiss=float(os.getenv("RW_W_DISMISS", "0.15")),
        wa3c_mu_dismissal=float(os.getenv("RW_WA3C_MU_DISMISSAL", "0.5")),
        throughput=float(os.getenv("RW_THROUGHPUT", "1.0")),
        wait_penalty=float(os.getenv("RW_WAIT_PENALTY", "0.038")),
        deadline_miss_penalty=float(os.getenv("RW_DEADLINE_MISS_PENALTY", "1.0")),
        energy_penalty=float(os.getenv("RW_ENERGY_PENALTY", "0.02")),
        priority_scale=float(os.getenv("RW_PRIORITY_SCALE", "0.065")),
        high_priority_miss_extra=float(os.getenv("RW_HIGH_PRIORITY_MISS_EXTRA", "3.4")),
        mid_priority_miss_extra=float(os.getenv("RW_MID_PRIORITY_MISS_EXTRA", "1.1")),
        queue_pressure_coef=float(os.getenv("RW_QUEUE_PRESSURE_COEF", "0.028")),
        miss_rate_stability_coef=float(os.getenv("RW_MISS_RATE_STABILITY_COEF", "0.14")),
    )
    jobs = load_jobs_csv(args.jobs_csv)
    return TraceSchedulingEnv(
        jobs=jobs,
        max_queue=args.max_queue,
        cluster_cores=args.cluster_cores,
        cluster_mem=args.cluster_mem,
        reward_weights=rw,
        max_steps=args.steps,
    )


def run_algo(algo: str, args: argparse.Namespace) -> Dict[str, str]:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(args)
    if algo in {"ppo", "a2c", "dqn"}:
        r = train_sb3_baseline(algo, env, args.steps, args.seed, out / "models")
        return {"algo": r.algo, "model_path": r.model_path, "steps": str(r.total_steps)}
    if algo == "sac_discrete":
        r = train_discrete_sac(env, args.steps, args.seed, out / "models")
        return {"algo": r.algo, "model_path": r.model_path, "steps": str(r.total_steps)}
    if algo == "decision_transformer":
        r = train_decision_transformer(env, args.steps, args.seed, out / "models")
        return {"algo": r.algo, "model_path": r.model_path, "steps": str(r.total_steps)}
    if algo == "iql_discrete":
        r = train_iql_discrete(env, args.steps, args.seed, out / "models")
        return {"algo": r.algo, "model_path": r.model_path, "steps": str(r.total_steps)}
    if algo == "cql_discrete":
        r = train_cql_discrete(env, args.steps, args.seed, out / "models")
        return {"algo": r.algo, "model_path": r.model_path, "steps": str(r.total_steps)}
    if algo == "deeprm_pg":
        r = train_deeprm_pg(env, args.steps, args.seed, out / "models")
        return {"algo": r.algo, "model_path": r.model_path, "steps": str(r.total_steps)}
    if algo == "decima_style":
        r = train_decima_style(env, args.steps, args.seed, out / "models")
        return {"algo": r.algo, "model_path": r.model_path, "steps": str(r.total_steps)}
    if algo == "deepjs_dqn":
        r = train_deepjs_dqn(env, args.steps, args.seed, out / "models")
        return {"algo": r.algo, "model_path": r.model_path, "steps": str(r.total_steps)}
    if algo == "performative_rl":
        r = train_performative_rl(env, args.steps, args.seed, out / "models")
        return {"algo": algo, "model_path": r.model_path, "steps": str(r.total_steps)}
    if algo == "hybrid_role_based_marl":
        r = train_hybrid_role(env, args.steps, args.seed, out / "models")
        return {"algo": algo, "model_path": r.model_path, "steps": str(r.total_steps)}
    raise ValueError(f"Unknown algo: {algo}")


def write_summary(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, str]] = []
    for algo in args.algos:
        print(f"Training {algo} ...", flush=True)
        rows.append(run_algo(algo, args))
    out = Path(args.out_dir)
    write_summary(out / "training_summary.csv", rows)
    print(f"Wrote {out / 'training_summary.csv'}")


if __name__ == "__main__":
    main()
