#!/usr/bin/env python3
"""
Train + evaluate all baselines, paper models, cloud baselines, and proposed models
on one real jobs.csv slice. Writes a ranked comparison CSV (composite score).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from collections import deque
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO

from benchmarks.data.job_dataset import load_jobs_csv
from benchmarks.envs.trace_scheduling_env import RewardWeights, TraceSchedulingEnv
from benchmarks.models.cloud_paper_models import PolicyValue as CloudPolicyValue
from benchmarks.models.cloud_paper_models import mlp as cloud_mlp
from benchmarks.models.paper_models import (
    DiscreteActor,
    DiscreteQ,
    TinyDecisionTransformer,
    mlp,
)
from benchmarks.models.proposed_models import PolicyValue as VanillaProposedPolicy
from benchmarks.models.proposed_models import RoleBasedPolicyValue
from benchmarks.models.proposed_models import _extract_role_weights, proposed_action_logits
from benchmarks.train_all import run_algo


def _progress_log(path: Path | None, line: str) -> None:
    """Append one timestamped line and flush (for tail -f)."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{ts} {line}\n")
        f.flush()


def _make_env(jobs_csv: str, max_queue: int, cluster_cores: float, cluster_mem: float, max_steps: int) -> TraceSchedulingEnv:
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
    jobs = load_jobs_csv(jobs_csv)
    return TraceSchedulingEnv(
        jobs=jobs,
        max_queue=max_queue,
        cluster_cores=cluster_cores,
        cluster_mem=cluster_mem,
        reward_weights=rw,
        max_steps=max_steps,
    )


def _rollout_metrics(env: TraceSchedulingEnv, act_fn: Callable[[np.ndarray], int], n_episodes: int = 3) -> Dict[str, float]:
    rets, miss, wait, ene, thr, fair = [], [], [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = trunc = False
        R = 0.0
        info: Dict = {}
        fair_steps: List[float] = []
        while not done and not trunc:
            a = act_fn(obs)
            obs, r, done, trunc, info = env.step(a)
            R += float(r)
            fair_steps.append(1.0 / (1.0 + float(env._fairness_variance_cpu())))
        rets.append(R)
        miss.append(float(info.get("deadline_misses", 0.0)))
        wait.append(float(info.get("mean_wait", 0.0)))
        ene.append(float(info.get("mean_energy", 0.0)))
        thr.append(float(info.get("completed", 0.0)) / max(float(info.get("time", 1.0)), 1.0))
        fair.append(float(np.mean(fair_steps)) if fair_steps else 1.0)
    return {
        "mean_return": float(np.mean(rets)),
        "mean_deadline_misses": float(np.mean(miss)),
        "mean_wait": float(np.mean(wait)),
        "mean_energy": float(np.mean(ene)),
        "mean_throughput": float(np.mean(thr)),
        "mean_fairness": float(np.mean(fair)),
    }


def _composite(m: Dict[str, float]) -> float:
    # Defaults aligned to SLA-sensitive scheduling objective used in publication sweeps.
    w_miss = float(os.getenv("COMP_W_MISS", "1.7"))
    w_wait = float(os.getenv("COMP_W_WAIT", "0.08"))
    w_energy = float(os.getenv("COMP_W_ENERGY", "0.05"))
    return (
        m["mean_return"]
        - w_miss * m["mean_deadline_misses"]
        - w_wait * m["mean_wait"]
        - w_energy * m["mean_energy"]
    )


def _eval_sb3(path: str, algo: str, env: TraceSchedulingEnv) -> Dict[str, float]:
    algo_l = algo.lower()
    if algo_l == "ppo":
        model = PPO.load(path)
    elif algo_l == "a2c":
        model = A2C.load(path)
    elif algo_l == "dqn":
        model = DQN.load(path)
    else:
        raise ValueError(algo)

    def act(obs: np.ndarray) -> int:
        a, _ = model.predict(obs, deterministic=True)
        return int(a)

    return _rollout_metrics(env, act)


def _eval_sac_actor(path: str, env: TraceSchedulingEnv) -> Dict[str, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    actor = DiscreteActor(obs_dim, n_act)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            logits = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            return int(torch.argmax(logits).item())

    return _rollout_metrics(env, act)


def _eval_cql(path: str, env: TraceSchedulingEnv) -> Dict[str, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    q = DiscreteQ(obs_dim, n_act)
    q.load_state_dict(ckpt["q"])
    q.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            qs = q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            return int(torch.argmax(qs).item())

    return _rollout_metrics(env, act)


def _eval_iql(path: str, env: TraceSchedulingEnv) -> Dict[str, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    pi = DiscreteActor(obs_dim, n_act)
    pi.load_state_dict(ckpt["pi"])
    pi.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            logits = pi(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            return int(torch.argmax(logits).item())

    return _rollout_metrics(env, act)


def _eval_dt(path: str, env: TraceSchedulingEnv, context_len: int = 20) -> Dict[str, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    model = TinyDecisionTransformer(obs_dim, n_act)
    model.load_state_dict(ckpt["model"])
    model.eval()
    hist_obs: deque = deque(maxlen=context_len)
    hist_rtg: deque = deque(maxlen=context_len)

    def act(obs: np.ndarray) -> int:
        hist_obs.append(obs.copy().astype(np.float32))
        hist_rtg.append(0.0)
        L = len(hist_obs)
        pad_obs = np.zeros((context_len, obs_dim), dtype=np.float32)
        pad_rtg = np.zeros((context_len, 1), dtype=np.float32)
        sl = np.stack(list(hist_obs), axis=0)
        pad_obs[-L:] = sl
        pad_rtg[-L:, 0] = np.asarray(list(hist_rtg), dtype=np.float32)
        x = np.concatenate([pad_obs, pad_rtg], axis=1)
        with torch.no_grad():
            logits = model(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(logits[0, -1]).item())

    return _rollout_metrics(env, act)


def _eval_deeprm(path: str, env: TraceSchedulingEnv) -> Dict[str, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    net = CloudPolicyValue(obs_dim, n_act)
    net.load_state_dict(ckpt["net"])
    net.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(logits).item())

    return _rollout_metrics(env, act)


def _eval_decima(path: str, env: TraceSchedulingEnv) -> Dict[str, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    scorer = cloud_mlp([obs_dim, 256, 256, n_act], act=torch.nn.ReLU)
    scorer.load_state_dict(ckpt["scorer"])
    scorer.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            logits = scorer(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            return int(torch.argmax(logits).item())

    return _rollout_metrics(env, act)


def _eval_deepjs(path: str, env: TraceSchedulingEnv) -> Dict[str, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    q = mlp([obs_dim, 256, 256, n_act], act=torch.nn.ReLU)
    q.load_state_dict(ckpt["q"])
    q.eval()

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            qs = q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            return int(torch.argmax(qs).item())

    return _rollout_metrics(env, act)


def _eval_proposed(path: str, env: TraceSchedulingEnv) -> Dict[str, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mode = ckpt["mode"]
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    if mode == "role":
        net = RoleBasedPolicyValue(obs_dim, n_act)
    else:
        net = VanillaProposedPolicy(obs_dim, n_act)
    net.load_state_dict(ckpt["net"])
    net.eval()

    role_sharp = float(ckpt.get("role_mix_sharp", 1.75)) if mode == "role" else 1.75
    env.reward_weights.proposed_softmax_beta = float(ckpt.get("priority_softmax_beta", 2.75))

    def act(obs: np.ndarray) -> int:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if mode == "role":
                rw = torch.as_tensor(
                    _extract_role_weights(obs, env.max_queue, env.slot_features, role_sharp),
                    dtype=torch.float32,
                ).unsqueeze(0)
                logits, _ = net(o, rw)
            else:
                logits, _ = net(o)
            adj = proposed_action_logits(env, obs, logits)
            return int(torch.argmax(adj.squeeze(0)).item())

    return _rollout_metrics(env, act)


def evaluate_trained(algo: str, model_path: str, env: TraceSchedulingEnv) -> Dict[str, float]:
    if algo in {"ppo", "a2c", "dqn"}:
        return _eval_sb3(model_path, algo, env)
    if algo == "sac_discrete":
        return _eval_sac_actor(model_path, env)
    if algo == "cql_discrete":
        return _eval_cql(model_path, env)
    if algo == "iql_discrete":
        return _eval_iql(model_path, env)
    if algo == "decision_transformer":
        return _eval_dt(model_path, env)
    if algo == "deeprm_pg":
        return _eval_deeprm(model_path, env)
    if algo == "decima_style":
        return _eval_decima(model_path, env)
    if algo == "deepjs_dqn":
        return _eval_deepjs(model_path, env)
    if algo in {"performative_rl", "hybrid_role_based_marl"}:
        return _eval_proposed(model_path, env)
    raise ValueError(f"No evaluator for {algo}")


# Unified real traces (repo-root relative paths).
REAL_DATASETS: List[Tuple[str, str]] = [
    ("datasets/real/google_cluster_2011/jobs.csv", "google_cluster_2011"),
    ("datasets/real/azure_vm_2019/jobs.csv", "azure_vm_2019"),
    ("datasets/real/alibaba_cluster_2018/jobs.csv", "alibaba_cluster_2018"),
]


DEFAULT_ALGOS: List[str] = [
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
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs_csv", type=str, default=None, help="Path to unified jobs.csv (ignored if --all_real).")
    ap.add_argument(
        "--all_real",
        action="store_true",
        help="Run on all three canonical real traces (Google, Azure, Alibaba).",
    )
    ap.add_argument("--dataset_name", type=str, default="real_slice", help="Label for output rows (single-run only).")
    ap.add_argument("--out_dir", type=str, default="benchmarks/outputs/full_suite_compare")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_queue", type=int, default=16)
    ap.add_argument("--cluster_cores", type=float, default=8.0)
    ap.add_argument("--cluster_mem", type=float, default=32.0)
    ap.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    ap.add_argument(
        "--progress_file",
        type=str,
        default=None,
        help="Append heartbeat lines here. Default: <out_dir>/progress.log",
    )
    ap.add_argument("--no_progress", action="store_true", help="Disable progress.log heartbeats.")
    return ap.parse_args()


def _run_one_dataset(
    jobs_csv: str,
    dataset_name: str,
    out: Path,
    seed: int,
    steps: int,
    max_queue: int,
    cluster_cores: float,
    cluster_mem: float,
    algos: List[str],
    progress_path: Path | None,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    n = len(algos)
    _progress_log(
        progress_path,
        f"BEGIN dataset={dataset_name} jobs_csv={jobs_csv} seed={seed} steps={steps} n_algos={n}",
    )

    ns = SimpleNamespace(
        jobs_csv=jobs_csv,
        out_dir=str(models_dir),
        seed=seed,
        steps=steps,
        max_queue=max_queue,
        cluster_cores=cluster_cores,
        cluster_mem=cluster_mem,
    )

    rows: List[Dict[str, str]] = []
    for i, algo in enumerate(algos, start=1):
        print(f"[{dataset_name}] Training {algo} ...", flush=True)
        _progress_log(progress_path, f"TRAIN_START {i}/{n} dataset={dataset_name} algo={algo}")
        sub = SimpleNamespace(**{**ns.__dict__, "out_dir": str(models_dir / algo)})
        train_row = run_algo(algo, sub)
        _progress_log(progress_path, f"EVAL_START {i}/{n} dataset={dataset_name} algo={algo}")
        env_eval = _make_env(jobs_csv, max_queue, cluster_cores, cluster_mem, steps)
        met = evaluate_trained(algo, train_row["model_path"], env_eval)
        comp = _composite(met)
        _progress_log(
            progress_path,
            f"DONE {i}/{n} dataset={dataset_name} algo={algo} "
            f"composite={comp:.6f} mean_return={met['mean_return']:.6f} "
            f"misses={met['mean_deadline_misses']:.4f} wait={met['mean_wait']:.4f}",
        )
        rows.append(
            {
                "dataset": dataset_name,
                "algo": algo,
                "model_path": train_row["model_path"],
                "composite_score": f"{comp:.6f}",
                "mean_return": f"{met['mean_return']:.6f}",
                "mean_deadline_misses": f"{met['mean_deadline_misses']:.6f}",
                "mean_wait": f"{met['mean_wait']:.6f}",
                "mean_energy": f"{met['mean_energy']:.6f}",
            }
        )

    rows.sort(key=lambda x: float(x["composite_score"]), reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = str(i + 1)
        r["is_best"] = "yes" if i == 0 else "no"

    fields = [
        "dataset",
        "rank",
        "algo",
        "composite_score",
        "mean_return",
        "mean_deadline_misses",
        "mean_wait",
        "mean_energy",
        "is_best",
        "model_path",
    ]
    csv_path = out / "full_suite_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if rows:
        best = rows[0]
        (out / "BEST_MODEL.txt").write_text(
            f"dataset={dataset_name}\n"
            f"best_algo={best['algo']}\n"
            f"composite_score={best['composite_score']}\n"
            f"mean_return={best['mean_return']}\n"
            f"model_path={best['model_path']}\n",
            encoding="utf-8",
        )
    print(f"Wrote {csv_path}", flush=True)
    _progress_log(progress_path, f"DATASET_COMPLETE dataset={dataset_name} csv={csv_path.as_posix()}")
    return csv_path


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    if args.all_real:
        base = Path(args.out_dir)
        base.mkdir(parents=True, exist_ok=True)
        if args.no_progress:
            progress_path: Path | None = None
        else:
            progress_path = Path(args.progress_file) if args.progress_file else base / "progress.log"
        _progress_log(
            progress_path,
            f"RUN_START all_real=True out_dir={base.as_posix()} seed={args.seed} steps={args.steps} "
            f"n_algos={len(args.algos)}",
        )
        all_rows: List[Dict[str, str]] = []
        for rel, name in REAL_DATASETS:
            jobs = (root / rel).as_posix()
            sub_out = base / name
            _run_one_dataset(
                jobs,
                name,
                sub_out,
                args.seed,
                args.steps,
                args.max_queue,
                args.cluster_cores,
                args.cluster_mem,
                args.algos,
                progress_path,
            )
            with (sub_out / "full_suite_comparison.csv").open(newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                all_rows.extend(list(r))
        agg = base / "full_suite_all_real.csv"
        if all_rows:
            fields = list(all_rows[0].keys())
            with agg.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(all_rows)
            print(f"Wrote aggregate {agg}", flush=True)
            _progress_log(progress_path, f"RUN_COMPLETE aggregate={agg.as_posix()}")
        else:
            _progress_log(progress_path, "RUN_COMPLETE (no aggregate rows)")
        return

    if not args.jobs_csv:
        raise SystemExit("Provide --jobs_csv or use --all_real.")

    out_single = Path(args.out_dir)
    if args.no_progress:
        progress_single: Path | None = None
    else:
        progress_single = Path(args.progress_file) if args.progress_file else out_single / "progress.log"
    _progress_log(
        progress_single,
        f"RUN_START all_real=False out_dir={out_single.as_posix()} dataset={args.dataset_name} "
        f"seed={args.seed} steps={args.steps} n_algos={len(args.algos)}",
    )
    _run_one_dataset(
        args.jobs_csv,
        args.dataset_name,
        out_single,
        args.seed,
        args.steps,
        args.max_queue,
        args.cluster_cores,
        args.cluster_mem,
        args.algos,
        progress_single,
    )
    _progress_log(progress_single, "RUN_COMPLETE")


if __name__ == "__main__":
    main()
