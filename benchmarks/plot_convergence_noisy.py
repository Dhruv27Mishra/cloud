#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO

from benchmarks.compare_full_suite import REAL_DATASETS, _make_env
from benchmarks.models.cloud_paper_models import PolicyValue as CloudPolicyValue
from benchmarks.models.cloud_paper_models import mlp as cloud_mlp
from benchmarks.models.paper_models import DiscreteActor, DiscreteQ, TinyDecisionTransformer, mlp
from benchmarks.models.proposed_models import (
    PolicyValue as VanillaProposedPolicy,
    RoleBasedPolicyValue,
    _extract_role_weights,
    proposed_action_logits,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compare_dir",
        type=str,
        default="benchmarks/outputs/full_real_wa3c_reward",
        help="Directory containing per-dataset full_suite_comparison.csv files.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="benchmarks/outputs/convergence_noisy",
        help="Output directory for plots and step-wise CSV.",
    )
    ap.add_argument("--jobs_csv", type=str, default=None, help="Single dataset jobs.csv path (optional).")
    ap.add_argument("--dataset_name", type=str, default=None, help="Name for single-dataset mode.")
    ap.add_argument(
        "--comparison_csv",
        type=str,
        default=None,
        help="Path to single dataset full_suite_comparison.csv (optional).",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Evaluation seeds for stochastic rollouts.",
    )
    ap.add_argument(
        "--plot_seed",
        type=int,
        default=0,
        help="Seed to visualize directly (no mean±std aggregation).",
    )
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--smooth_window", type=int, default=1, help="Rolling window (1 = raw values).")
    ap.add_argument("--max_queue", type=int, default=16)
    ap.add_argument("--cluster_cores", type=float, default=8.0)
    ap.add_argument("--cluster_mem", type=float, default=32.0)
    return ap.parse_args()


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_act_fn(algo: str, model_path: str, env) -> Callable[[np.ndarray], int]:
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n

    if algo in {"ppo", "a2c", "dqn"}:
        if algo == "ppo":
            model = PPO.load(model_path)
        elif algo == "a2c":
            model = A2C.load(model_path)
        else:
            model = DQN.load(model_path)

        def act(obs: np.ndarray) -> int:
            a, _ = model.predict(obs, deterministic=False)
            return int(a)

        return act

    if algo == "sac_discrete":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        actor = DiscreteActor(obs_dim, n_act)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()

        def act(obs: np.ndarray) -> int:
            with torch.no_grad():
                logits = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                dist = torch.distributions.Categorical(logits=logits)
                return int(dist.sample().item())

        return act

    if algo == "cql_discrete":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        q = DiscreteQ(obs_dim, n_act)
        q.load_state_dict(ckpt["q"])
        q.eval()

        def act(obs: np.ndarray) -> int:
            with torch.no_grad():
                qs = q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                temp = 0.25
                dist = torch.distributions.Categorical(logits=qs / temp)
                return int(dist.sample().item())

        return act

    if algo == "iql_discrete":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        pi = DiscreteActor(obs_dim, n_act)
        pi.load_state_dict(ckpt["pi"])
        pi.eval()

        def act(obs: np.ndarray) -> int:
            with torch.no_grad():
                logits = pi(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                dist = torch.distributions.Categorical(logits=logits)
                return int(dist.sample().item())

        return act

    if algo == "decision_transformer":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model = TinyDecisionTransformer(obs_dim, n_act)
        model.load_state_dict(ckpt["model"])
        model.eval()
        hist_obs: deque = deque(maxlen=20)
        hist_rtg: deque = deque(maxlen=20)

        def act(obs: np.ndarray) -> int:
            hist_obs.append(obs.copy().astype(np.float32))
            hist_rtg.append(0.0)
            L = len(hist_obs)
            pad_obs = np.zeros((20, obs_dim), dtype=np.float32)
            pad_rtg = np.zeros((20, 1), dtype=np.float32)
            sl = np.stack(list(hist_obs), axis=0)
            pad_obs[-L:] = sl
            pad_rtg[-L:, 0] = np.asarray(list(hist_rtg), dtype=np.float32)
            x = np.concatenate([pad_obs, pad_rtg], axis=1)
            with torch.no_grad():
                logits = model(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0))[0, -1]
                dist = torch.distributions.Categorical(logits=logits)
                return int(dist.sample().item())

        return act

    if algo == "deeprm_pg":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        net = CloudPolicyValue(obs_dim, n_act)
        net.load_state_dict(ckpt["net"])
        net.eval()

        def act(obs: np.ndarray) -> int:
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits.squeeze(0))
                return int(dist.sample().item())

        return act

    if algo == "decima_style":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        scorer = cloud_mlp([obs_dim, 256, 256, n_act], act=torch.nn.ReLU)
        scorer.load_state_dict(ckpt["scorer"])
        scorer.eval()

        def act(obs: np.ndarray) -> int:
            with torch.no_grad():
                logits = scorer(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                dist = torch.distributions.Categorical(logits=logits)
                return int(dist.sample().item())

        return act

    if algo == "deepjs_dqn":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        q = mlp([obs_dim, 256, 256, n_act], act=torch.nn.ReLU)
        q.load_state_dict(ckpt["q"])
        q.eval()

        def act(obs: np.ndarray) -> int:
            with torch.no_grad():
                qs = q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                temp = 0.25
                dist = torch.distributions.Categorical(logits=qs / temp)
                return int(dist.sample().item())

        return act

    if algo in {"performative_rl", "hybrid_role_based_marl"}:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        mode = ckpt["mode"]
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
                dist = torch.distributions.Categorical(logits=adj.squeeze(0))
                return int(dist.sample().item())

        return act

    raise ValueError(f"Unknown algo: {algo}")


def _run_step_trace(env, act_fn: Callable[[np.ndarray], int], seed: int) -> Dict[str, List[float]]:
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    rewards: List[float] = []
    throughputs: List[float] = []
    fairness: List[float] = []
    while not done and not trunc:
        a = act_fn(obs)
        obs, r, done, trunc, info = env.step(a)
        rewards.append(float(r))
        throughputs.append(float(info.get("completed", 0.0)) / max(float(info.get("time", 1.0)), 1.0))
        fairness.append(float(env._fairness_variance_cpu()))
    return {"reward": rewards, "throughput": throughputs, "fairness": fairness}


def _mean_std_curves(traces: List[List[float]], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.full((len(traces), max_len), np.nan, dtype=np.float64)
    for i, t in enumerate(traces):
        arr[i, : len(t)] = np.asarray(t, dtype=np.float64)
    valid = np.sum(~np.isnan(arr), axis=0)
    mean = np.zeros((max_len,), dtype=np.float64)
    std = np.zeros((max_len,), dtype=np.float64)
    for j in range(max_len):
        if valid[j] == 0:
            mean[j] = np.nan
            std[j] = np.nan
        elif valid[j] == 1:
            vals = arr[~np.isnan(arr[:, j]), j]
            mean[j] = float(vals[0])
            std[j] = 0.0
        else:
            vals = arr[~np.isnan(arr[:, j]), j]
            mean[j] = float(np.mean(vals))
            std[j] = float(np.std(vals))
    return mean, std


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    w = np.ones((window,), dtype=np.float64) / float(window)
    valid = ~np.isnan(y)
    y0 = np.where(valid, y, 0.0)
    num = np.convolve(y0, w, mode="same")
    den = np.convolve(valid.astype(np.float64), w, mode="same")
    out = np.full_like(num, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 1e-12)
    return out


def _plot_metric(metric: str, grouped: Dict[str, List[List[float]]], out_png: Path, smooth_window: int, trace_idx: int) -> None:
    plt.figure(figsize=(12, 7))
    plt.style.use("seaborn-v0_8-whitegrid")
    title = {
        "reward": "Reward convergence",
        "throughput": "Throughput convergence",
        "fairness": "Fairness convergence",
    }.get(metric, f"Convergence ({metric})")

    ylabels = {
        "reward": "Reward (unitless score)",
        "throughput": "Throughput (completed jobs per simulation-time unit)",
        "fairness": "Fairness proxy: variance of queued CPU demand (CPU^2)",
    }

    for algo in sorted(grouped.keys()):
        if len(grouped[algo]) <= trace_idx:
            continue
        raw = np.asarray(grouped[algo][trace_idx], dtype=np.float64)
        xs = np.arange(1, len(raw) + 1)
        sm = _smooth(raw, smooth_window)
        plt.plot(xs, sm, linewidth=2.7, label=algo)
    plt.xlabel("Training episodes")
    plt.ylabel(ylabels.get(metric, metric.title()))
    plt.title(title)
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=170)
    plt.close()


def main() -> None:
    args = parse_args()
    compare_dir = Path(args.compare_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_step_rows: List[Dict[str, str]] = []

    if args.jobs_csv:
        dataset_specs = [(args.jobs_csv, args.dataset_name or "synthetic")]
    else:
        dataset_specs = [
            (str((Path(__file__).resolve().parents[1] / rel_jobs).as_posix()), dataset) for rel_jobs, dataset in REAL_DATASETS
        ]

    for jobs_csv, dataset in dataset_specs:
        if args.comparison_csv:
            csv_path = Path(args.comparison_csv)
        else:
            csv_path = compare_dir / dataset / "full_suite_comparison.csv"
        rows = _load_rows(csv_path)
        grouped: Dict[str, Dict[str, List[List[float]]]] = defaultdict(lambda: defaultdict(list))

        for row in rows:
            algo = row["algo"]
            model_path = row["model_path"]
            for s in args.seeds:
                env = _make_env(jobs_csv, args.max_queue, args.cluster_cores, args.cluster_mem, args.max_steps)
                act_fn = _build_act_fn(algo, model_path, env)
                trace = _run_step_trace(env, act_fn, s)
                grouped["reward"][algo].append(trace["reward"])
                grouped["throughput"][algo].append(trace["throughput"])
                grouped["fairness"][algo].append(trace["fairness"])
                for i, (rv, tv, fv) in enumerate(zip(trace["reward"], trace["throughput"], trace["fairness"]), start=1):
                    all_step_rows.append(
                        {
                            "dataset": dataset,
                            "algo": algo,
                            "seed": str(s),
                            "step": str(i),
                            "reward": f"{rv:.8f}",
                            "throughput": f"{tv:.8f}",
                            "fairness": f"{fv:.8f}",
                        }
                    )

        d_out = out_root / dataset
        try:
            trace_idx = list(args.seeds).index(args.plot_seed)
        except ValueError:
            trace_idx = 0

        _plot_metric(
            "reward",
            grouped["reward"],
            d_out / "convergence_reward_noisy.png",
            smooth_window=args.smooth_window,
            trace_idx=trace_idx,
        )
        _plot_metric(
            "throughput",
            grouped["throughput"],
            d_out / "convergence_throughput_noisy.png",
            smooth_window=args.smooth_window,
            trace_idx=trace_idx,
        )
        _plot_metric(
            "fairness",
            grouped["fairness"],
            d_out / "convergence_fairness_noisy.png",
            smooth_window=args.smooth_window,
            trace_idx=trace_idx,
        )

    csv_out = out_root / "convergence_stepwise_all.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "algo", "seed", "step", "reward", "throughput", "fairness"])
        w.writeheader()
        w.writerows(all_step_rows)
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()

