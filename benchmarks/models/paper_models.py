from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.envs.trace_scheduling_env import TraceSchedulingEnv


def mlp(sizes: List[int], act=nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


@dataclass
class TrainResult:
    algo: str
    total_steps: int
    model_path: str


class ReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int = 200_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i] = o
        self.actions[i] = a
        self.rewards[i] = r
        self.next_obs[i] = no
        self.dones[i] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.actions[idx]),
            torch.as_tensor(self.rewards[idx]),
            torch.as_tensor(self.next_obs[idx]),
            torch.as_tensor(self.dones[idx]),
        )


class DiscreteActor(nn.Module):
    def __init__(self, obs_dim: int, n_act: int):
        super().__init__()
        self.net = mlp([obs_dim, 256, 256, n_act], act=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DiscreteQ(nn.Module):
    def __init__(self, obs_dim: int, n_act: int):
        super().__init__()
        self.net = mlp([obs_dim, 256, 256, n_act], act=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _collect_random_warmup(env: TraceSchedulingEnv, rb: ReplayBuffer, steps: int = 3000) -> None:
    obs, _ = env.reset()
    for _ in range(steps):
        a = env.action_space.sample()
        no, r, d, t, _ = env.step(a)
        rb.add(obs, a, r, no, float(d or t))
        obs = no
        if d or t:
            obs, _ = env.reset()


def train_discrete_sac(
    env: TraceSchedulingEnv,
    total_steps: int,
    seed: int,
    out_dir: str | Path,
) -> TrainResult:
    """
    SAC discrete variant (Christodoulou-style approximation).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    actor = DiscreteActor(obs_dim, n_act)
    q1 = DiscreteQ(obs_dim, n_act)
    q2 = DiscreteQ(obs_dim, n_act)
    tq1 = DiscreteQ(obs_dim, n_act)
    tq2 = DiscreteQ(obs_dim, n_act)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())

    a_opt = optim.Adam(actor.parameters(), lr=3e-4)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)

    rb = ReplayBuffer(obs_dim=obs_dim, capacity=200_000)
    _collect_random_warmup(env, rb, steps=3000)
    gamma = 0.99
    tau = 0.01
    alpha = 0.2

    obs, _ = env.reset()
    for step in range(total_steps):
        with torch.no_grad():
            logits = actor(torch.as_tensor(obs).unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            a = int(np.random.choice(np.arange(n_act), p=probs))
        no, r, d, t, _ = env.step(a)
        rb.add(obs, a, r, no, float(d or t))
        obs = no
        if d or t:
            obs, _ = env.reset()

        if rb.size < 4096:
            continue
        b_obs, b_act, b_rew, b_nobs, b_done = rb.sample(256)
        with torch.no_grad():
            next_logits = actor(b_nobs)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_logp = torch.log(next_probs + 1e-8)
            tq = torch.min(tq1(b_nobs), tq2(b_nobs))
            v_next = (next_probs * (tq - alpha * next_logp)).sum(dim=-1)
            y = b_rew + gamma * (1.0 - b_done) * v_next

        q1_pred = q1(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
        q2_pred = q2(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
        q_loss = (q1_pred - y).pow(2).mean() + (q2_pred - y).pow(2).mean()
        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        logits = actor(b_obs)
        probs = torch.softmax(logits, dim=-1)
        logp = torch.log(probs + 1e-8)
        q_min = torch.min(q1(b_obs), q2(b_obs))
        a_loss = (probs * (alpha * logp - q_min)).sum(dim=-1).mean()
        a_opt.zero_grad()
        a_loss.backward()
        a_opt.step()

        if step % 2 == 0:
            for p, tp in zip(q1.parameters(), tq1.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
            for p, tp in zip(q2.parameters(), tq2.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    ckpt = out / "sac_discrete.pt"
    torch.save({"actor": actor.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict()}, ckpt)
    return TrainResult(algo="sac_discrete", total_steps=total_steps, model_path=ckpt.as_posix())


def train_iql_discrete(
    env: TraceSchedulingEnv,
    total_steps: int,
    seed: int,
    out_dir: str | Path,
) -> TrainResult:
    """
    IQL-style (Kostrikov et al., 2022) approximation for discrete actions.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    q = DiscreteQ(obs_dim, n_act)
    v = mlp([obs_dim, 256, 256, 1], act=nn.ReLU)
    pi = DiscreteActor(obs_dim, n_act)
    q_opt = optim.Adam(q.parameters(), lr=3e-4)
    v_opt = optim.Adam(v.parameters(), lr=3e-4)
    pi_opt = optim.Adam(pi.parameters(), lr=3e-4)

    rb = ReplayBuffer(obs_dim=obs_dim, capacity=200_000)
    _collect_random_warmup(env, rb, steps=6000)
    gamma = 0.99
    expectile = 0.7
    beta = 3.0

    for _ in range(total_steps):
        b_obs, b_act, b_rew, b_nobs, b_done = rb.sample(256)
        with torch.no_grad():
            target = b_rew + gamma * (1.0 - b_done) * v(b_nobs).squeeze(1)
        q_sa = q(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
        q_loss = (q_sa - target).pow(2).mean()
        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        with torch.no_grad():
            q_det = q(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
        v_pred = v(b_obs).squeeze(1)
        diff = q_det - v_pred
        w = torch.where(diff > 0, expectile, 1.0 - expectile)
        v_loss = (w * diff.pow(2)).mean()
        v_opt.zero_grad()
        v_loss.backward()
        v_opt.step()

        with torch.no_grad():
            adv = q(b_obs) - v(b_obs)
            weights = torch.exp(beta * adv).clamp(max=100.0)
            target_probs = torch.softmax(weights, dim=-1)
        log_probs = torch.log_softmax(pi(b_obs), dim=-1)
        pi_loss = -(target_probs * log_probs).sum(dim=-1).mean()
        pi_opt.zero_grad()
        pi_loss.backward()
        pi_opt.step()

    ckpt = out / "iql_discrete.pt"
    torch.save({"q": q.state_dict(), "v": v.state_dict(), "pi": pi.state_dict()}, ckpt)
    return TrainResult(algo="iql_discrete", total_steps=total_steps, model_path=ckpt.as_posix())


class TinyDecisionTransformer(nn.Module):
    def __init__(self, obs_dim: int, n_act: int, d_model: int = 128, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim + 1, d_model)  # obs + RTG token
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d_model, n_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.obs_proj(x)
        h = self.tr(z)
        return self.head(h)


def _build_offline_buffer(env: TraceSchedulingEnv, steps: int = 20_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_list = []
    act_list = []
    rew_list = []
    obs, _ = env.reset()
    for _ in range(steps):
        a = env.action_space.sample()
        no, r, d, t, _ = env.step(a)
        obs_list.append(obs.copy())
        act_list.append(a)
        rew_list.append(r)
        obs = no
        if d or t:
            obs, _ = env.reset()
    return np.asarray(obs_list), np.asarray(act_list), np.asarray(rew_list, dtype=np.float32)


def train_decision_transformer(
    env: TraceSchedulingEnv,
    total_steps: int,
    seed: int,
    out_dir: str | Path,
    context_len: int = 20,
) -> TrainResult:
    """
    Decision Transformer style (Chen et al., 2021) compact implementation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    obs, acts, rews = _build_offline_buffer(env, steps=max(15_000, total_steps))
    rtg = np.flip(np.cumsum(np.flip(rews))).astype(np.float32)

    model = TinyDecisionTransformer(obs_dim, n_act)
    opt = optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    n = len(obs)
    for _ in range(total_steps):
        i = np.random.randint(context_len, n)
        sl = slice(i - context_len, i)
        x_obs = torch.as_tensor(obs[sl], dtype=torch.float32)
        x_rtg = torch.as_tensor(rtg[sl], dtype=torch.float32).unsqueeze(1)
        x = torch.cat([x_obs, x_rtg], dim=1).unsqueeze(0)
        y = torch.as_tensor(acts[sl], dtype=torch.long).unsqueeze(0)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, n_act), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    ckpt = out / "decision_transformer.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    return TrainResult(algo="decision_transformer", total_steps=total_steps, model_path=ckpt.as_posix())


def train_crossq_style_discrete(
    env: TraceSchedulingEnv,
    total_steps: int,
    seed: int,
    out_dir: str | Path,
) -> TrainResult:
    """
    CrossQ-style (Bhatt et al., 2024 inspired) pragmatic discrete adaptation:
    dual critics, aggressive update ratio, target smoothing and critic agreement regularization.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    q1 = DiscreteQ(obs_dim, n_act)
    q2 = DiscreteQ(obs_dim, n_act)
    tq1 = DiscreteQ(obs_dim, n_act)
    tq2 = DiscreteQ(obs_dim, n_act)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=4e-4)
    rb = ReplayBuffer(obs_dim, capacity=250_000)
    _collect_random_warmup(env, rb, steps=5000)

    gamma = 0.99
    tau = 0.02
    reg = 0.01
    obs, _ = env.reset()
    for step in range(total_steps):
        with torch.no_grad():
            q = q1(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
            if np.random.rand() < max(0.02, 0.2 * (1 - step / max(1, total_steps))):
                a = env.action_space.sample()
            else:
                a = int(np.argmax(q))
        no, r, d, t, _ = env.step(a)
        rb.add(obs, a, r, no, float(d or t))
        obs = no
        if d or t:
            obs, _ = env.reset()

        if rb.size < 4096:
            continue
        for _ in range(2):  # aggressive critic updates
            b_obs, b_act, b_rew, b_nobs, b_done = rb.sample(256)
            with torch.no_grad():
                next_q = torch.min(tq1(b_nobs), tq2(b_nobs))
                next_a = torch.argmax(next_q, dim=1)
                target = b_rew + gamma * (1 - b_done) * next_q.gather(1, next_a.unsqueeze(1)).squeeze(1)
            q1_sa = q1(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            q2_sa = q2(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            agree = (q1_sa - q2_sa).pow(2).mean()
            loss = (q1_sa - target).pow(2).mean() + (q2_sa - target).pow(2).mean() + reg * agree
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % 2 == 0:
            for p, tp in zip(q1.parameters(), tq1.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
            for p, tp in zip(q2.parameters(), tq2.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    ckpt = out / "crossq_style_discrete.pt"
    torch.save({"q1": q1.state_dict(), "q2": q2.state_dict()}, ckpt)
    return TrainResult(algo="crossq_style_discrete", total_steps=total_steps, model_path=ckpt.as_posix())


def train_cql_discrete(
    env: TraceSchedulingEnv,
    total_steps: int,
    seed: int,
    out_dir: str | Path,
) -> TrainResult:
    """
    Conservative Q-Learning style discrete variant (Kumar et al., NeurIPS 2020 inspired).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    q = DiscreteQ(obs_dim, n_act)
    tq = DiscreteQ(obs_dim, n_act)
    tq.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=3e-4)

    # Offline-style dataset from behavior policy trajectories.
    rb = ReplayBuffer(obs_dim, capacity=250_000)
    _collect_random_warmup(env, rb, steps=max(20_000, total_steps))

    gamma = 0.99
    tau = 0.01
    cql_alpha = 1.0

    for step in range(total_steps):
        b_obs, b_act, b_rew, b_nobs, b_done = rb.sample(256)
        with torch.no_grad():
            next_q = tq(b_nobs)
            next_v = torch.max(next_q, dim=1).values
            target = b_rew + gamma * (1.0 - b_done) * next_v

        q_all = q(b_obs)
        q_data = q_all.gather(1, b_act.unsqueeze(1)).squeeze(1)
        bellman = (q_data - target).pow(2).mean()
        conservative = (torch.logsumexp(q_all, dim=1) - q_data).mean()
        loss = bellman + cql_alpha * conservative

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2 == 0:
            for p, tp in zip(q.parameters(), tq.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    ckpt = out / "cql_discrete.pt"
    torch.save({"q": q.state_dict()}, ckpt)
    return TrainResult(algo="cql_discrete", total_steps=total_steps, model_path=ckpt.as_posix())
