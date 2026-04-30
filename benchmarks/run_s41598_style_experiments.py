#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.compare_full_suite import _make_env
from benchmarks.plot_convergence_noisy import _build_act_fn


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="benchmarks/outputs/s41598_reproduction")
    ap.add_argument("--steps_train", type=int, default=1200)
    ap.add_argument("--max_eps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comp_w_miss", type=float, default=1.7)
    ap.add_argument("--comp_w_wait", type=float, default=0.08)
    ap.add_argument("--comp_w_energy", type=float, default=0.05)
    ap.add_argument("--proposed_shape_scale", type=float, default=7.0)
    ap.add_argument("--hybrid_fair_gain_coef", type=float, default=1.6)
    ap.add_argument("--hybrid_fair_bonus_coef", type=float, default=0.22)
    ap.add_argument("--hybrid_lr_base", type=float, default=1e-3)
    ap.add_argument("--performative_lr", type=float, default=8e-4)
    ap.add_argument("--performative_lambda", type=float, default=0.005)
    ap.add_argument("--hybrid_performative_lambda", type=float, default=0.0025)
    return ap.parse_args()


def _run_compare(jobs_csv: str, out_dir: Path, steps_train: int, seed: int, algos: List[str], env_extra: Dict[str, str]) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["PROPOSED_FAST"] = "1"
    env.update(env_extra)
    cmd = [
        "python3",
        "benchmarks/compare_full_suite.py",
        "--jobs_csv",
        jobs_csv,
        "--dataset_name",
        out_dir.name,
        "--out_dir",
        str(out_dir),
        "--steps",
        str(steps_train),
        "--seed",
        str(seed),
        "--algos",
        *algos,
    ]
    subprocess.run(cmd, check=True, env=env)
    return out_dir / "full_suite_comparison.csv"


def _read_rows(p: Path) -> List[Dict[str, str]]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rollout_metrics(jobs_csv: str, algo: str, model_path: str, max_eps: int) -> Dict[str, List[float]]:
    env = _make_env(jobs_csv, max_queue=16, cluster_cores=8.0, cluster_mem=32.0, max_steps=max_eps)
    act = _build_act_fn(algo, model_path, env)
    obs, _ = env.reset(seed=0)
    done = trunc = False
    rew, rej, lbf = [], [], []
    while not done and not trunc and len(rew) < max_eps:
        a = act(obs)
        obs, r, done, trunc, info = env.step(a)
        arrived = max(int(getattr(env, "idx", 0)), 1)
        rej_rate = 100.0 * float(info.get("dismissals_total", 0.0)) / float(arrived)
        rew.append(float(r))
        rej.append(rej_rate)
        lbf.append(float(env._fairness_variance_cpu()))
    return {"reward": rew, "task_rejection_rate_pct": rej, "load_balance_factor": lbf}


def _smooth(y: np.ndarray, w: int = 45) -> np.ndarray:
    if w <= 1:
        return y
    k = np.ones((w,), dtype=np.float64) / float(w)
    return np.convolve(y, k, mode="same")


def _plot_convergence(curves: Dict[str, List[float]], title: str, ylabel: str, out: Path) -> None:
    plt.figure(figsize=(9.2, 4.8))
    plt.style.use("seaborn-v0_8-whitegrid")
    pal = plt.get_cmap("tab20")
    for i, (algo, y) in enumerate(curves.items()):
        ys = _smooth(np.asarray(y, dtype=np.float64), 45)
        lw = 2.6 if algo == "hybrid_role_based_marl" else (2.2 if algo == "a2c" else 1.5)
        plt.plot(np.arange(1, len(ys) + 1), ys, label=algo, color=pal(i % 20), linewidth=lw)
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), dpi=400, bbox_inches="tight")
    plt.close()


def _plot_bar(values: Dict[str, float], title: str, ylabel: str, higher_better: bool, out: Path) -> None:
    items = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_better)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(9.2, 4.8))
    cols = ["#1f77b4" if a == "hybrid_role_based_marl" else ("#2ca02c" if a == "a2c" else "#b0b0b0") for a in labels]
    plt.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.4)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), dpi=400, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    base_env = {
        "COMP_W_MISS": str(args.comp_w_miss),
        "COMP_W_WAIT": str(args.comp_w_wait),
        "COMP_W_ENERGY": str(args.comp_w_energy),
        "PROPOSED_SHAPE_SCALE": str(args.proposed_shape_scale),
        "HYBRID_FAIR_GAIN_COEF": str(args.hybrid_fair_gain_coef),
        "HYBRID_FAIR_BONUS_COEF": str(args.hybrid_fair_bonus_coef),
        "HYBRID_LR": str(args.hybrid_lr_base),
        "PERFORMATIVE_LR": str(args.performative_lr),
        "PERFORMATIVE_LAMBDA": str(args.performative_lambda),
        "HYBRID_PERFORMATIVE_LAMBDA": str(args.hybrid_performative_lambda),
        "HYBRID_USE_Q_GUIDANCE": "0",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Same experimental format: convergence sensitivity + baseline comparisons.
    algos = [
        "a2c",
        "dqn",
        "ppo",
        "sac_discrete",
        "iql_discrete",
        "decision_transformer",
        "deeprm_pg",
        "decima_style",
        "deepjs_dqn",
        "performative_rl",
        "hybrid_role_based_marl",
    ]
    cmp_csv = _run_compare(args.jobs_csv, out / "main_compare", args.steps_train, args.seed, algos, base_env)
    rows = _read_rows(cmp_csv)

    # Fig-like convergence plots for reward/rejection/load-balance.
    by_algo = {}
    for r in rows:
        by_algo[r["algo"]] = _rollout_metrics(args.jobs_csv, r["algo"], r["model_path"], args.max_eps)
    for metric, ylab, ttl in [
        ("reward", "Reward value", "Convergence curve (reward)"),
        ("task_rejection_rate_pct", "Task rejection rate (%)", "Convergence curve (task rejection rate)"),
        ("load_balance_factor", "Load-balance factor", "Convergence curve (load-balance factor)"),
    ]:
        curves = {a: m[metric] for a, m in by_algo.items()}
        _plot_convergence(curves, ttl, ylab, out / f"s41598_{metric}_convergence")
        finals = {a: float(v[-1]) for a, v in curves.items() if len(v) > 0}
        _plot_bar(finals, f"Comparison across algorithms ({metric})", ylab, metric == "reward", out / f"s41598_{metric}_bar")

    # LR sensitivity (paper-style convergence sensitivity experiment).
    lr_vals = [3e-4, 5e-4, 7e-4, 1e-3]
    lr_curves = {}
    for lr in lr_vals:
        lr_dir = out / f"lr_{lr:.0e}".replace("+0", "")
        p = _run_compare(
            args.jobs_csv,
            lr_dir,
            args.steps_train,
            args.seed,
            ["hybrid_role_based_marl"],
            {**base_env, "HYBRID_LR": str(lr)},
        )
        rr = _read_rows(p)[0]
        m = _rollout_metrics(args.jobs_csv, "hybrid_role_based_marl", rr["model_path"], args.max_eps)
        lr_curves[f"lr={lr:.0e}"] = m["reward"]
    _plot_convergence(lr_curves, "Convergence with different learning rates", "Reward value", out / "s41598_lr_sensitivity_convergence")
    print(f"Wrote S41598-style experiment outputs to {out}")


if __name__ == "__main__":
    main()

