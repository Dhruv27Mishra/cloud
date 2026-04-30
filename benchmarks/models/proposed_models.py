from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.envs.trace_scheduling_env import TraceSchedulingEnv, priority_tier


def proposed_action_logits(env: TraceSchedulingEnv, obs: np.ndarray, logits: torch.Tensor) -> torch.Tensor:
    """
    Reward-aligned action shaping for proposed models.
    Combines learned logits with a one-step heuristic score aligned to env reward terms.
    """
    rw = env.reward_weights
    beta = float(getattr(rw, "proposed_softmax_beta", 2.75))
    mq, sf = env.max_queue, env.slot_features
    qi = mq * sf + 1
    qlen = int(round(float(obs[qi])))
    qlen = max(0, min(mq, qlen))
    if logits.dim() == 2:
        L = logits.squeeze(0).clone()
    else:
        L = logits.clone()

    p_max_slot = max((float(obs[i * sf + 3]) for i in range(qlen)), default=1.0)
    p_scale = max(p_max_slot, float(getattr(env, "_p_max", 1.0)), 1e-6)
    l_max = max(float(getattr(env, "_l_max", 1.0)), 1e-6)
    tau_max = max(float(getattr(env, "_tau_max", 1.0)), 1e-6)

    # Adaptive blend: rely more on heuristic in high-pressure/urgent states.
    min_slack = min((float(obs[i * sf + 5]) for i in range(qlen)), default=0.0)
    q_frac = float(qlen) / float(max(1, mq))
    urgent = 1.0 if min_slack < 0.0 else 0.0
    alpha = float(np.clip(0.45 + 0.35 * q_frac + 0.25 * urgent, 0.45, 0.95))
    scale = float(os.getenv("PROPOSED_SHAPE_SCALE", "8.0"))
    use_shape = os.getenv("HYBRID_USE_ACTION_SHAPING", "1") == "1"
    shape_coef = float(os.getenv("HYBRID_ACTION_BONUS_COEF", "1.0"))
    fair_gain_coef = float(os.getenv("HYBRID_FAIR_GAIN_COEF", "1.4"))

    cpu_list = [float(obs[i * sf + 1]) for i in range(qlen)]
    base_var = float(np.var(np.asarray(cpu_list, dtype=np.float32))) if qlen > 1 else 0.0

    for a in range(qlen):
        off = a * sf
        runtime = float(obs[off + 0])
        cpu = float(obs[off + 1])
        mem = float(obs[off + 2])
        prio = float(obs[off + 3])
        wait = float(obs[off + 4])
        slack = float(obs[off + 5])
        p_norm = float(np.clip(prio / p_scale, 0.0, 1.0))

        # One-step proxy terms
        lat = max(0.0, wait + runtime)
        lat_norm = float(np.clip(lat / l_max, 0.0, 1.0))
        wait_norm = float(np.clip(wait / l_max, 0.0, 1.5))
        qos = 1.0 - lat_norm
        prio_term = p_norm * (1.0 - lat_norm)
        energy = (0.5 * cpu + 0.3 * mem + 0.2 * min(1.0, runtime / 100.0)) * (runtime / tau_max)

        # Fairness gain from serving this slot now (variance reduction proxy)
        if qlen > 1:
            rem = cpu_list[:a] + cpu_list[a + 1 :]
            rem_var = float(np.var(np.asarray(rem, dtype=np.float32))) if rem else 0.0
            fair_gain = base_var - rem_var
        else:
            fair_gain = 0.0

        urgency = float(np.clip((-slack) / max(runtime, 1e-6), 0.0, 2.0))
        hopeless = float(np.clip((-(slack + 0.5 * runtime)) / max(runtime, 1e-6), 0.0, 2.0))
        runtime_cost = float(np.clip(runtime / max(tau_max, 1e-6), 0.0, 1.5))

        if bool(getattr(rw, "use_wa3c", False)):
            h = (
                rw.w_qos * qos
                + rw.w_priority * prio_term
                - rw.w_energy * energy
                + rw.w_fair * fair_gain_coef * fair_gain
                + 0.20 * urgency
                - 0.24 * hopeless
                - 0.12 * wait_norm
                - 0.22 * runtime_cost
            )
        else:
            h = (
                rw.throughput
                - rw.wait_penalty * wait
                - rw.energy_penalty * energy
                + rw.priority_scale * p_norm
                + 0.22 * urgency
                + 0.15 * fair_gain_coef * fair_gain
                - 0.18 * hopeless
                - 0.10 * wait_norm
                - 0.18 * runtime_cost
            )

        shaped = scale * h + beta * p_norm
        if use_shape:
            L[a] = (1.0 - alpha) * L[a] + alpha * (shape_coef * shaped)
        else:
            L[a] = L[a] + beta * p_norm
    for a in range(qlen, mq):
        L[a] = -1e9
    return L.unsqueeze(0) if logits.dim() == 2 else L


def proposed_action_logits_batch(env: TraceSchedulingEnv, obs_np: np.ndarray, logits: torch.Tensor) -> torch.Tensor:
    rows = []
    for bi in range(obs_np.shape[0]):
        rows.append(proposed_action_logits(env, obs_np[bi], logits[bi : bi + 1]).squeeze(0))
    return torch.stack(rows, dim=0)


def mlp(sizes: List[int], act=nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


@dataclass
class TrainEvalResult:
    algo: str
    model_path: str
    mean_return: float
    mean_deadline_misses: float
    mean_wait: float
    mean_energy: float
    total_steps: int = 0


class PolicyValue(nn.Module):
    """Shared actor–critic for performative RL (single policy, no role heads)."""

    def __init__(self, obs_dim: int, n_act: int, hidden: int = 512):
        super().__init__()
        self.pi = mlp([obs_dim, hidden, hidden, hidden, n_act], act=nn.ReLU)
        self.v = mlp([obs_dim, hidden, hidden, hidden, 1], act=nn.ReLU)

    def forward(self, obs: torch.Tensor):
        return self.pi(obs), self.v(obs).squeeze(-1)


class RoleBasedPolicyValue(nn.Module):
    """Role-based policy: three small MLP heads merged by queue priority-class weights."""

    def __init__(self, obs_dim: int, n_act: int, hidden: int = 512, head_hidden: int = 320):
        super().__init__()
        self.encoder = mlp([obs_dim, hidden, hidden, hidden], act=nn.ReLU)
        self.enc_ln = nn.LayerNorm(hidden)
        self.head_lo = mlp([hidden, head_hidden, n_act], act=nn.ReLU)
        self.head_md = mlp([hidden, head_hidden, n_act], act=nn.ReLU)
        self.head_hi = mlp([hidden, head_hidden, n_act], act=nn.ReLU)
        self.v = nn.Linear(hidden, 1)
        self.q1 = mlp([hidden, head_hidden, n_act], act=nn.ReLU)
        self.q2 = mlp([hidden, head_hidden, n_act], act=nn.ReLU)

    def forward(self, obs: torch.Tensor, role_w: torch.Tensor):
        h = self.enc_ln(self.encoder(obs))
        logits = (
            role_w[:, 0:1] * self.head_lo(h)
            + role_w[:, 1:2] * self.head_md(h)
            + role_w[:, 2:3] * self.head_hi(h)
        )
        return logits, self.v(h).squeeze(-1)

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.enc_ln(self.encoder(obs))
        q1 = self.q1(h)
        q2 = self.q2(h)
        return torch.minimum(q1, q2)

    def q_values_pair(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_ln(self.encoder(obs))
        return self.q1(h), self.q2(h)


def _extract_role_weights(
    obs: np.ndarray,
    max_queue: int,
    slot_features: int,
    role_mix_sharp: float = 1.75,
) -> np.ndarray:
    """Mix role heads using only occupied queue slots (exclude padded zeros)."""
    qi = max_queue * slot_features + 1
    qlen = int(round(float(obs[qi])))
    qlen = max(0, min(max_queue, qlen))
    pri: List[int] = []
    for i in range(qlen):
        p = float(obs[i * slot_features + 3])
        pri.append(priority_tier(p))
    if not pri:
        return np.ones(3, dtype=np.float32) / 3.0
    cnt = np.bincount(np.asarray(pri, dtype=np.int64), minlength=3).astype(np.float32)
    w = cnt / max(1.0, float(cnt.sum()))
    sharp = float(role_mix_sharp)
    w = np.power(np.maximum(w, 1e-8), sharp)
    w = w / w.sum()
    return w.astype(np.float32)


def _compute_gae(rew: np.ndarray, done: np.ndarray, values: np.ndarray, gamma: float = 0.99, lam: float = 0.95):
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        delta = rew[t] + gamma * values[t + 1] * (1.0 - done[t]) - values[t]
        g = delta + gamma * lam * (1.0 - done[t]) * g
        adv[t] = g
    ret = adv + values[:-1]
    return adv, ret


def _hybrid_action_shape_bonus(obs: np.ndarray, action: int, max_queue: int, slot_features: int) -> float:
    """
    Role-only dense shaping:
    - reward selecting urgent / high-priority slots
    - penalize regret vs best urgent candidate in queue
    """
    qi = max_queue * slot_features + 1
    qlen = int(round(float(obs[qi])))
    qlen = max(0, min(max_queue, qlen))
    if qlen <= 0 or action < 0 or action >= qlen:
        return 0.0

    p_max = max((float(obs[i * slot_features + 3]) for i in range(qlen)), default=1.0)
    p_scale = max(p_max, 1.0)
    fair_coef = float(os.getenv("HYBRID_FAIR_BONUS_COEF", "0.22"))
    scores: List[float] = []
    for i in range(qlen):
        off = i * slot_features
        rt = max(float(obs[off + 0]), 1e-6)
        pr = float(obs[off + 3]) / p_scale
        wt = float(obs[off + 4])
        sl = float(obs[off + 5])
        urg = float(np.clip((-sl) / rt, 0.0, 2.0))
        hopeless = float(np.clip((-(sl + 0.5 * rt)) / rt, 0.0, 2.0))
        wait_norm = float(np.clip(wt / (rt * 4.0), 0.0, 2.0))
        rt_cost = float(np.clip(rt / 20.0, 0.0, 2.0))
        score = 1.45 * pr + 1.05 * urg + 0.34 * wait_norm - 0.55 * hopeless - 0.32 * rt_cost
        scores.append(score)

    chosen = scores[action]
    best = max(scores)
    bonus = 0.22 * chosen - 0.16 * (best - chosen)
    # Encourage load-balance improvement: reward actions that reduce queue CPU variance.
    cpus = [float(obs[i * slot_features + 1]) for i in range(qlen)]
    if len(cpus) > 1:
        base_var = float(np.var(np.asarray(cpus, dtype=np.float32)))
        rem = cpus[:action] + cpus[action + 1 :]
        rem_var = float(np.var(np.asarray(rem, dtype=np.float32))) if rem else 0.0
        bonus += fair_coef * (base_var - rem_var)

    # Extra SLA credit for taking a truly urgent high-priority slot.
    off = action * slot_features
    rt = max(float(obs[off + 0]), 1e-6)
    pr = float(obs[off + 3]) / p_scale
    sl = float(obs[off + 5])
    if pr >= 0.67 and sl < 0.0:
        hopeless = float(np.clip((-(sl + 0.5 * rt)) / rt, 0.0, 2.0))
        bonus += 0.24 * float(np.clip((-sl) / rt, 0.0, 2.0)) - 0.18 * hopeless
    return float(bonus)


def _evaluate(policy_fn, env: TraceSchedulingEnv, episodes: int = 4) -> Dict[str, float]:
    rets, miss, wait, ene = [], [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = trunc = False
        R = 0.0
        info = {}
        while not done and not trunc:
            a = int(policy_fn(obs))
            obs, r, done, trunc, info = env.step(a)
            R += float(r)
        rets.append(R)
        miss.append(float(info.get("deadline_misses", 0.0)))
        wait.append(float(info.get("mean_wait", 0.0)))
        ene.append(float(info.get("mean_energy", 0.0)))
    return {
        "mean_return": float(np.mean(rets)),
        "mean_deadline_misses": float(np.mean(miss)),
        "mean_wait": float(np.mean(wait)),
        "mean_energy": float(np.mean(ene)),
    }


def _train_on_policy(
    env: TraceSchedulingEnv,
    net_mode: str,
    total_steps: int,
    seed: int,
    performative_lambda: float,
    out_dir: str | Path,
) -> TrainEvalResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_act = env.action_space.n
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if net_mode == "role":
        net = RoleBasedPolicyValue(obs_dim, n_act)
        # Hybrid-only: more rollouts / entropy so role heads specialize under SLA-shaped rewards.
        lr = float(os.getenv("HYBRID_LR", "7e-4"))
        rollout = 1280
        n_epochs = 10
        clip = 0.11
        ent_coef = 0.042
        role_mix_sharp = float(os.getenv("HYBRID_ROLE_MIX_SHARP", "2.05"))
    else:
        net = PolicyValue(obs_dim, n_act)
        lr = float(os.getenv("PERFORMATIVE_LR", "7e-4"))
        rollout = 1280
        n_epochs = 10
        clip = 0.11
        ent_coef = 0.034
        role_mix_sharp = 1.75

    opt = optim.Adam(net.parameters(), lr=lr)
    mb = 64
    vf_coef = 0.5
    max_grad = 0.5
    gamma = 0.99
    q_coef = float(os.getenv("HYBRID_Q_COEF", "0.52"))
    cql_alpha = float(os.getenv("HYBRID_CQL_ALPHA", "0.5"))
    bc_coef = float(os.getenv("HYBRID_BC_COEF", "0.08"))
    use_q_aux = os.getenv("HYBRID_USE_Q_AUX", "1") == "1"
    use_q_guidance = os.getenv("HYBRID_USE_Q_GUIDANCE", "0") == "1"
    completion_bonus_coef = float(os.getenv("HYBRID_COMPLETION_BONUS_COEF", "0.12"))

    obs, _ = env.reset(seed=seed)
    mu_hat = 1.0
    done_steps = 0
    while done_steps < total_steps:
        O, NO, A, R, D, LP, V, Aux = [], [], [], [], [], [], [], []
        for _ in range(rollout):
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if net_mode == "role":
                rw = _extract_role_weights(obs, env.max_queue, env.slot_features, role_mix_sharp)
                rw_t = torch.as_tensor(rw, dtype=torch.float32).unsqueeze(0)
                logits, v = net(o, rw_t)
                aux_item = rw
            else:
                logits, v = net(o)
                aux_item = np.zeros((1,), dtype=np.float32)
            adj = proposed_action_logits(env, obs, logits)
            dist = torch.distributions.Categorical(logits=adj.squeeze(0))
            a = dist.sample()
            lp = dist.log_prob(a)
            no, r, d, t, info = env.step(int(a.item()))
            if net_mode == "role":
                r = float(r) + _hybrid_action_shape_bonus(obs, int(a.item()), env.max_queue, env.slot_features)
                # Small completion/throughput bonus to reward actionable progress under congestion.
                rate = float(info.get("completed", 0.0)) / max(float(info.get("time", 1.0)), 1.0)
                r += completion_bonus_coef * rate
            q_pressure = float(len(env.queue)) / float(max(1, env.max_queue))
            miss_pressure = float(info.get("deadline_misses", 0.0)) / float(max(1, env.completed))
            mu_hat = 0.98 * mu_hat + 0.02 * (1.0 + q_pressure + miss_pressure)
            if performative_lambda > 0:
                r = float(r) - performative_lambda * (mu_hat - 1.0) ** 2
            O.append(obs.copy())
            NO.append(no.copy())
            A.append(int(a.item()))
            R.append(float(r))
            D.append(float(d or t))
            LP.append(float(lp.item()))
            V.append(float(v.item()))
            Aux.append(aux_item)
            obs = no
            done_steps += 1
            if d or t:
                obs, _ = env.reset()
            if done_steps >= total_steps:
                break
        if not O:
            break
        with torch.no_grad():
            o_last = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            if net_mode == "role":
                rw = _extract_role_weights(obs, env.max_queue, env.slot_features, role_mix_sharp)
                _, v_last = net(o_last, torch.as_tensor(rw, dtype=torch.float32).unsqueeze(0))
            else:
                _, v_last = net(o_last)
            v_last = float(v_last.item())

        rewards = np.asarray(R, dtype=np.float32)
        dones = np.asarray(D, dtype=np.float32)
        values = np.asarray(V + [v_last], dtype=np.float32)
        adv, ret = _compute_gae(rewards, dones, values)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_all = np.asarray(O, dtype=np.float32)
        nobs_all = np.asarray(NO, dtype=np.float32)
        act_all = np.asarray(A, dtype=np.int64)
        rew_all = np.asarray(R, dtype=np.float32)
        done_all = np.asarray(D, dtype=np.float32)
        lp_old = torch.as_tensor(np.asarray(LP, dtype=np.float32))
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        idx = np.arange(len(obs_all))
        for _ in range(n_epochs):
            np.random.shuffle(idx)
            for st in range(0, len(idx), mb):
                j = idx[st : st + mb]
                o_b = torch.as_tensor(obs_all[j], dtype=torch.float32)
                a_b = torch.as_tensor(act_all[j], dtype=torch.long)
                lp_old_b = lp_old[j]
                ret_b = ret_t[j]
                adv_b = adv_t[j]
                if net_mode == "role":
                    rw_b = torch.as_tensor(np.asarray([Aux[k] for k in j], dtype=np.float32))
                    logits, v_b = net(o_b, rw_b)
                else:
                    logits, v_b = net(o_b)
                adj = proposed_action_logits_batch(env, obs_all[j], logits)
                dist = torch.distributions.Categorical(logits=adj)
                lp = dist.log_prob(a_b)
                ratio = torch.exp(lp - lp_old_b)
                s1 = ratio * adv_b
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_b
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = (v_b - ret_b).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pi_loss + vf_coef * v_loss - ent_coef * ent
                if net_mode == "role" and use_q_aux:
                    # CQL-inspired conservative value guidance for hybrid head.
                    q1_all, q2_all = net.q_values_pair(o_b)
                    q1_data = q1_all.gather(1, a_b.unsqueeze(1)).squeeze(1)
                    q2_data = q2_all.gather(1, a_b.unsqueeze(1)).squeeze(1)
                    b_rew = torch.as_tensor(rew_all[j], dtype=torch.float32)
                    b_done = torch.as_tensor(done_all[j], dtype=torch.float32)
                    b_nobs = torch.as_tensor(nobs_all[j], dtype=torch.float32)
                    with torch.no_grad():
                        q_next = net.q_values(b_nobs)
                        next_v = torch.max(q_next, dim=1).values
                        q_target = b_rew + gamma * (1.0 - b_done) * next_v
                    q_td = (q1_data - q_target).pow(2).mean() + (q2_data - q_target).pow(2).mean()
                    conservative = (
                        (torch.logsumexp(q1_all, dim=1) - q1_data).mean()
                        + (torch.logsumexp(q2_all, dim=1) - q2_data).mean()
                    ) * 0.5

                    # Pressure-aware adaptation: increase conservatism under heavier queue pressure.
                    q_idx = env.max_queue * env.slot_features + 1
                    q_frac_b = torch.as_tensor(obs_all[j, q_idx] / max(1.0, float(env.max_queue)), dtype=torch.float32)
                    press = torch.clamp(torch.mean(q_frac_b), 0.0, 1.0)
                    q_coef_eff = q_coef * (1.0 + 0.45 * press.item())
                    cql_eff = cql_alpha * (1.0 + 0.60 * press.item())

                    # Advantage-weighted behavior cloning regularizer (stabilizes good actions).
                    adv_pos = torch.clamp(adv_b.detach(), min=0.0)
                    denom = torch.mean(adv_pos) + 1e-6
                    bc_loss = -torch.mean(dist.log_prob(a_b) * (adv_pos / denom))
                    loss = loss + q_coef_eff * q_td + cql_eff * conservative + bc_coef * bc_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad)
                opt.step()

    stem = f"{net_mode}_pl{performative_lambda:.3f}".replace(".", "p")
    ckpt = out / f"{stem}.pt"
    ckpt_payload: Dict = {"net": net.state_dict(), "mode": net_mode, "performative_lambda": performative_lambda}
    if net_mode == "role":
        ckpt_payload["role_mix_sharp"] = role_mix_sharp
    ckpt_payload["priority_softmax_beta"] = float(getattr(env.reward_weights, "proposed_softmax_beta", 2.75))
    torch.save(ckpt_payload, ckpt)

    def _policy(obs_np: np.ndarray) -> int:
        with torch.no_grad():
            o = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            if net_mode == "role":
                rw = torch.as_tensor(
                    _extract_role_weights(obs_np, env.max_queue, env.slot_features, role_mix_sharp),
                    dtype=torch.float32,
                ).unsqueeze(0)
                logits, _ = net(o, rw)
                if use_q_guidance:
                    qg = net.q_values(o)
                    qg = (qg - qg.mean(dim=1, keepdim=True)) / (qg.std(dim=1, keepdim=True) + 1e-6)
                    logits = logits + 0.55 * qg
            else:
                logits, _ = net(o)
            adj = proposed_action_logits(env, obs_np, logits)
            return int(torch.argmax(adj.squeeze(0)).item())

    m = _evaluate(_policy, env, episodes=4)
    return TrainEvalResult(
        algo=stem,
        model_path=ckpt.as_posix(),
        mean_return=m["mean_return"],
        mean_deadline_misses=m["mean_deadline_misses"],
        mean_wait=m["mean_wait"],
        mean_energy=m["mean_energy"],
        total_steps=total_steps,
    )


def _proposed_env_step_budget(requested: int, mult: int = 7) -> int:
    """On-policy PPO needs more env transitions than typical SB3/off-policy budgets."""
    # Fast-mode for quick iteration: disable step multiplication.
    if os.getenv("PROPOSED_FAST", "0") == "1":
        return int(requested)
    return max(int(requested), int(requested) * mult)


def train_performative_rl(env: TraceSchedulingEnv, total_steps: int, seed: int, out_dir: str | Path) -> TrainEvalResult:
    eff = _proposed_env_step_budget(total_steps)
    perf_lam = float(os.getenv("PERFORMATIVE_LAMBDA", "0.006"))
    return _train_on_policy(env, "vanilla", eff, seed, perf_lam, out_dir)


def train_hybrid_role(env: TraceSchedulingEnv, total_steps: int, seed: int, out_dir: str | Path) -> TrainEvalResult:
    # Hybrid benefits from extra on-policy data vs vanilla performative.
    eff = _proposed_env_step_budget(total_steps, mult=12)
    hybrid_lam = float(os.getenv("HYBRID_PERFORMATIVE_LAMBDA", "0.003"))
    return _train_on_policy(env, "role", eff, seed, hybrid_lam, out_dir)
