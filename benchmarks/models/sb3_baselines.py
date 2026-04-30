from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from benchmarks.envs.trace_scheduling_env import TraceSchedulingEnv


@dataclass
class TrainResult:
    algo: str
    total_steps: int
    model_path: str


def _make_vec(env: TraceSchedulingEnv) -> DummyVecEnv:
    return DummyVecEnv([lambda: Monitor(env)])


def train_sb3_baseline(
    algo: str,
    env: TraceSchedulingEnv,
    total_steps: int,
    seed: int,
    out_dir: str | Path,
) -> TrainResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vec = _make_vec(env)
    algo_l = algo.lower()

    if algo_l == "ppo":
        model = PPO("MlpPolicy", vec, verbose=0, seed=seed, learning_rate=3e-4, n_steps=256, batch_size=64)
    elif algo_l == "a2c":
        model = A2C("MlpPolicy", vec, verbose=0, seed=seed, learning_rate=7e-4)
    elif algo_l == "dqn":
        model = DQN(
            "MlpPolicy",
            vec,
            verbose=0,
            seed=seed,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=128,
        )
    elif algo_l == "sac":
        # SAC in SB3 supports Box action spaces only. For discrete scheduling action spaces,
        # keep this placeholder so you can swap in a discrete SAC implementation if needed.
        raise ValueError("SB3 SAC requires continuous actions; use discrete SAC implementation for this env.")
    else:
        raise ValueError(f"Unknown baseline algo: {algo}")

    model.learn(total_timesteps=total_steps)
    path = out / f"{algo_l}_model.zip"
    model.save(path.as_posix())
    vec.close()
    return TrainResult(algo=algo_l, total_steps=total_steps, model_path=path.as_posix())


def evaluate_policy_greedy(model, env: TraceSchedulingEnv, n_episodes: int = 3) -> Dict[str, float]:
    rewards = []
    misses = []
    waits = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        ret = 0.0
        info = {}
        while not done and not trunc:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(int(action))
            ret += r
        rewards.append(ret)
        misses.append(float(info.get("deadline_misses", 0.0)))
        waits.append(float(info.get("mean_wait", 0.0)))
    return {
        "mean_return": float(sum(rewards) / max(1, len(rewards))),
        "mean_deadline_misses": float(sum(misses) / max(1, len(misses))),
        "mean_wait": float(sum(waits) / max(1, len(waits))),
    }
