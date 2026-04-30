from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

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


class PolicyValue(nn.Module):
    def __init__(self, obs_dim: int, n_act: int):
        super().__init__()
        self.pi = mlp([obs_dim, 256, 256, n_act], act=nn.Tanh)
        self.v = mlp([obs_dim, 256, 256, 1], act=nn.Tanh)

    def forward(self, obs: torch.Tensor):
        return self.pi(obs), self.v(obs).squeeze(-1)


def train_deeprm_pg(
    env: TraceSchedulingEnv,
    total_steps: int,
    seed: int,
    out_dir: str | Path,
) -> TrainResult:
    """
    DeepRM-style policy gradient baseline (Mao et al., HotNets 2016 / arXiv 2016 inspired).
    Practical actor-critic adaptation for this discrete scheduling env.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    net = PolicyValue(obs_dim, n_act)
    opt = optim.Adam(net.parameters(), lr=3e-4)
    gamma = 0.99
    ent_coef = 0.01
    vf_coef = 0.5
    rollout = 256

    obs, _ = env.reset(seed=seed)
    steps = 0
    while steps < total_steps:
        o_buf, a_buf, r_buf, d_buf, lp_buf, v_buf = [], [], [], [], [], []
        for _ in range(rollout):
            o = torch.as_tensor(obs, dtype=torch.float32)
            logits, v = net(o)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            lp = dist.log_prob(a)
            no, r, d, t, _ = env.step(int(a.item()))
            o_buf.append(obs)
            a_buf.append(int(a.item()))
            r_buf.append(float(r))
            d_buf.append(float(d or t))
            lp_buf.append(lp)
            v_buf.append(v)
            obs = no
            steps += 1
            if d or t:
                obs, _ = env.reset()
            if steps >= total_steps:
                break

        if not o_buf:
            break
        with torch.no_grad():
            _, v_last = net(torch.as_tensor(obs, dtype=torch.float32))
            v_last = float(v_last.item())

        rets = []
        g = v_last
        for r, d in zip(reversed(r_buf), reversed(d_buf)):
            g = r + gamma * g * (1.0 - d)
            rets.append(g)
        rets = torch.as_tensor(list(reversed(rets)), dtype=torch.float32)
        vals = torch.stack(v_buf)
        adv = rets - vals.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logps = torch.stack(lp_buf)
        entropy = torch.distributions.Categorical(
            logits=net(torch.as_tensor(np.asarray(o_buf), dtype=torch.float32))[0]
        ).entropy().mean()
        pi_loss = -(logps * adv).mean()
        v_loss = (rets - vals).pow(2).mean()
        loss = pi_loss + vf_coef * v_loss - ent_coef * entropy
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

    ckpt = out / "deeprm_pg.pt"
    torch.save({"net": net.state_dict()}, ckpt)
    return TrainResult(algo="deeprm_pg", total_steps=total_steps, model_path=ckpt.as_posix())


def train_decima_style(
    env: TraceSchedulingEnv,
    total_steps: int,
    seed: int,
    out_dir: str | Path,
) -> TrainResult:
    """
    Decima-style scheduling policy (Mao et al., SIGCOMM 2019 inspired).
    This is a compact queue-attention actor-critic approximation for non-DAG benchmark envs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # "Attention-like" scorer on flattened observation (practical approximation).
    scorer = mlp([obs_dim, 256, 256, n_act], act=nn.ReLU)
    value = mlp([obs_dim, 256, 256, 1], act=nn.ReLU)
    opt = optim.Adam(list(scorer.parameters()) + list(value.parameters()), lr=2.5e-4)
    gamma = 0.99
    ent_coef = 0.01
    vf_coef = 0.5

    obs, _ = env.reset(seed=seed)
    steps = 0
    while steps < total_steps:
        o_buf, lp_buf, r_buf, d_buf, v_buf = [], [], [], [], []
        for _ in range(256):
            o = torch.as_tensor(obs, dtype=torch.float32)
            logits = scorer(o)
            v = value(o).squeeze(0)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            lp = dist.log_prob(a)
            no, r, d, t, _ = env.step(int(a.item()))
            o_buf.append(obs)
            lp_buf.append(lp)
            r_buf.append(float(r))
            d_buf.append(float(d or t))
            v_buf.append(v)
            obs = no
            steps += 1
            if d or t:
                obs, _ = env.reset()
            if steps >= total_steps:
                break
        if not o_buf:
            break
        with torch.no_grad():
            v_last = float(value(torch.as_tensor(obs, dtype=torch.float32)).item())
        rets = []
        g = v_last
        for r, d in zip(reversed(r_buf), reversed(d_buf)):
            g = r + gamma * g * (1.0 - d)
            rets.append(g)
        rets = torch.as_tensor(list(reversed(rets)), dtype=torch.float32)
        vals = torch.stack(v_buf).reshape(-1)
        adv = (rets - vals.detach())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pi_loss = -(torch.stack(lp_buf) * adv).mean()
        v_loss = (rets - vals).pow(2).mean()
        ent = torch.distributions.Categorical(
            logits=scorer(torch.as_tensor(np.asarray(o_buf), dtype=torch.float32))
        ).entropy().mean()
        loss = pi_loss + vf_coef * v_loss - ent_coef * ent
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(scorer.parameters()) + list(value.parameters()), 0.5)
        opt.step()

    ckpt = out / "decima_style.pt"
    torch.save({"scorer": scorer.state_dict(), "value": value.state_dict()}, ckpt)
    return TrainResult(algo="decima_style", total_steps=total_steps, model_path=ckpt.as_posix())


def train_deepjs_dqn(
    env: TraceSchedulingEnv,
    total_steps: int,
    seed: int,
    out_dir: str | Path,
) -> TrainResult:
    """
    DeepJS-style DQN baseline (cloud scheduling DRL line, 2019-era inspired).
    Lightweight DQN tailored to queue scheduling.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    q = mlp([obs_dim, 256, 256, n_act], act=nn.ReLU)
    tq = mlp([obs_dim, 256, 256, n_act], act=nn.ReLU)
    tq.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=1e-4)
    gamma = 0.99
    tau = 0.01
    cap = 100_000
    batch = 128

    obs_b = np.zeros((cap, obs_dim), dtype=np.float32)
    act_b = np.zeros((cap,), dtype=np.int64)
    rew_b = np.zeros((cap,), dtype=np.float32)
    nobs_b = np.zeros((cap, obs_dim), dtype=np.float32)
    don_b = np.zeros((cap,), dtype=np.float32)
    ptr = 0
    size = 0

    obs, _ = env.reset(seed=seed)
    for step in range(total_steps):
        eps = max(0.02, 0.25 * (1 - step / max(1, total_steps)))
        if np.random.rand() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(torch.argmax(q(torch.as_tensor(obs, dtype=torch.float32))).item())
        no, r, d, t, _ = env.step(a)
        obs_b[ptr] = obs
        act_b[ptr] = a
        rew_b[ptr] = r
        nobs_b[ptr] = no
        don_b[ptr] = float(d or t)
        ptr = (ptr + 1) % cap
        size = min(size + 1, cap)
        obs = no
        if d or t:
            obs, _ = env.reset()

        if size < 3000:
            continue
        idx = np.random.randint(0, size, size=batch)
        bo = torch.as_tensor(obs_b[idx], dtype=torch.float32)
        ba = torch.as_tensor(act_b[idx], dtype=torch.long)
        br = torch.as_tensor(rew_b[idx], dtype=torch.float32)
        bno = torch.as_tensor(nobs_b[idx], dtype=torch.float32)
        bd = torch.as_tensor(don_b[idx], dtype=torch.float32)
        with torch.no_grad():
            y = br + gamma * (1.0 - bd) * torch.max(tq(bno), dim=1).values
        qv = q(bo).gather(1, ba.unsqueeze(1)).squeeze(1)
        loss = (qv - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 2 == 0:
            for p, tp in zip(q.parameters(), tq.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    ckpt = out / "deepjs_dqn.pt"
    torch.save({"q": q.state_dict()}, ckpt)
    return TrainResult(algo="deepjs_dqn", total_steps=total_steps, model_path=ckpt.as_posix())
