from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from benchmarks.data.job_dataset import Job


def priority_tier(priority: float) -> int:
    """Discrete class 0/1/2; matches benchmarks.models.proposed_models._extract_role_weights."""
    if priority <= 0.0:
        return 0
    if priority < 2.0:
        return 1
    return 2


@dataclass
class RewardWeights:
    """If use_wa3c is True, step reward follows weighted WA3C-style components (Eq. reward)."""

    use_wa3c: bool = True
    # Eq. weighted sum; sum_i w_i = 1 (paper defaults)
    w_qos: float = 0.25
    w_energy: float = 0.2
    w_priority: float = 0.25
    w_fair: float = 0.15
    w_dismiss: float = 0.15
    # Energy model (paper)
    wa3c_W_base: float = 100.0
    wa3c_W_max: float = 200.0
    wa3c_CPI: float = 1.0
    wa3c_MAPI: float = 1.0
    wa3c_z_iota: float = 0.3
    wa3c_alpha_energy: float = 1.0
    wa3c_lambda_fair: float = 0.12
    wa3c_mu_dismissal: float = 0.5
    # Priority-weighted softmax bias on policy logits (proposed trainers only)
    proposed_softmax_beta: float = 2.75

    # --- legacy (used only when use_wa3c is False) ---
    throughput: float = 1.0
    wait_penalty: float = 0.038
    deadline_miss_penalty: float = 1.0
    energy_penalty: float = 0.02
    priority_scale: float = 0.065
    role_on_time_bonus: Tuple[float, float, float] = (0.42, 0.62, 0.98)
    tight_slack_on_time_bonus: float = 0.52
    high_priority_miss_extra: float = 3.4
    mid_priority_miss_extra: float = 1.1
    queue_pressure_coef: float = 0.028
    miss_rate_stability_coef: float = 0.14


class TraceSchedulingEnv(gym.Env):
    """
    Discrete scheduling env over a finite job list.
    Action: pick one slot from queue [0..max_queue-1].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        jobs: List[Job],
        max_queue: int = 16,
        cluster_cores: float = 8.0,
        cluster_mem: float = 32.0,
        reward_weights: RewardWeights | None = None,
        max_steps: int = 5000,
    ) -> None:
        super().__init__()
        self.jobs = jobs
        self.max_queue = max_queue
        self.cluster_cores = cluster_cores
        self.cluster_mem = cluster_mem
        self.reward_weights = reward_weights or RewardWeights()
        self.max_steps = max_steps

        self.slot_features = 6
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10_000.0,
            shape=(self.max_queue * self.slot_features + 5,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_queue)
        if self.jobs:
            self._l_max = float(max(max(1.0, j.deadline_time - j.arrival_time) for j in self.jobs))
            self._p_max = float(max(float(j.priority) for j in self.jobs))
            self._tau_max = float(max(j.runtime for j in self.jobs))
        else:
            self._l_max = self._p_max = self._tau_max = 1.0
        self._energy_scale = (self.reward_weights.wa3c_W_max * max(self._tau_max, 1e-6)) + 1e-6
        self._reset_state()

    def _reset_state(self) -> None:
        self.t = 0.0
        self.step_count = 0
        self.idx = 0
        self.queue: Deque[Job] = deque()
        self.completed = 0
        self.deadline_misses = 0
        self.total_wait = 0.0
        self.total_energy = 0.0
        self.dismissals_total = 0
        self._fill_arrivals()

    def _fairness_variance_cpu(self) -> float:
        cpus = [float(j.cpu_demand) for j in self.queue]
        if len(cpus) < 2:
            return 0.0
        return float(np.var(np.asarray(cpus, dtype=np.float64)))

    def _fill_arrivals(self) -> int:
        """Admit arrivals up to current time; count jobs dropped when queue is full (dismissals)."""
        dismissed = 0
        while self.idx < len(self.jobs) and self.jobs[self.idx].arrival_time <= self.t:
            if len(self.queue) < self.max_queue:
                self.queue.append(self.jobs[self.idx])
            else:
                dismissed += 1
                self.dismissals_total += 1
            self.idx += 1
        return dismissed

    def _slot_obs(self, j: Job) -> List[float]:
        wait = max(0.0, self.t - j.arrival_time)
        slack = j.deadline_time - self.t
        return [j.runtime, j.cpu_demand, j.mem_demand, float(j.priority), wait, slack]

    def _obs(self) -> np.ndarray:
        out: List[float] = []
        q_list = list(self.queue)
        for k in range(self.max_queue):
            if k < len(q_list):
                out.extend(self._slot_obs(q_list[k]))
            else:
                out.extend([0.0] * self.slot_features)
        out.extend(
            [
                self.t,
                float(len(self.queue)),
                float(self.completed),
                float(self.deadline_misses),
                float(self.idx) / max(1.0, float(len(self.jobs))),
            ]
        )
        return np.asarray(out, dtype=np.float32)

    def _pick_job(self, action: int) -> Tuple[Job | None, bool]:
        if len(self.queue) == 0:
            return None, False
        if action < 0 or action >= len(self.queue):
            return None, False
        q_list = list(self.queue)
        job = q_list[action]
        del q_list[action]
        self.queue = deque(q_list)
        return job, True

    def _wa3c_reward_completion(self, job: Job, finish_t: float, miss: int) -> float:
        rw = self.reward_weights
        L_j = finish_t - job.arrival_time
        R_qos = 1.0 - min(1.0, L_j / self._l_max)
        if miss:
            R_qos -= 0.35

        U_cpu = float(np.clip(job.cpu_demand / max(self.cluster_cores, 1e-6), 0.0, 1.0))
        iota = rw.wa3c_CPI * rw.wa3c_z_iota + rw.wa3c_MAPI * (1.0 - rw.wa3c_z_iota)
        tau = job.runtime
        e_raw = (rw.wa3c_W_base + U_cpu * iota * (rw.wa3c_W_max - rw.wa3c_W_base)) * tau
        e_norm = e_raw / self._energy_scale
        R_e = -rw.wa3c_alpha_energy * e_norm

        p_norm = float(job.priority) / max(self._p_max, 1e-6)
        p_norm = float(np.clip(p_norm, 0.0, 1.0))
        L_norm = min(1.0, L_j / self._l_max)
        R_p = p_norm * (1.0 - L_norm)

        R_f = -rw.wa3c_lambda_fair * self._fairness_variance_cpu()

        return (
            rw.w_qos * R_qos
            + rw.w_energy * R_e
            + rw.w_priority * R_p
            + rw.w_fair * R_f
        )

    def _wa3c_reward_idle(self) -> float:
        rw = self.reward_weights
        R_f = -rw.wa3c_lambda_fair * self._fairness_variance_cpu()
        return rw.w_fair * R_f - 0.02

    def reset(self, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int):
        self.step_count += 1
        job, valid = self._pick_job(int(action))
        reward = 0.0
        rw = self.reward_weights

        if job is not None and valid:
            finish_t = self.t + job.runtime
            miss = 1 if finish_t > job.deadline_time else 0
            wait = max(0.0, self.t - job.arrival_time)
            energy = 0.5 * job.cpu_demand + 0.3 * job.mem_demand + 0.2 * min(1.0, job.runtime / 100.0)

            self.completed += 1
            self.deadline_misses += miss
            self.total_wait += wait
            self.total_energy += energy

            if rw.use_wa3c:
                reward = self._wa3c_reward_completion(job, finish_t, miss)
            else:
                tier = priority_tier(float(job.priority))
                miss_f = float(miss)
                slack_at_sched = job.deadline_time - self.t
                reward = (
                    rw.throughput
                    - rw.wait_penalty * wait
                    - rw.deadline_miss_penalty * miss_f
                    - rw.energy_penalty * energy
                    + rw.priority_scale * float(job.priority)
                )
                if miss == 0:
                    reward += rw.role_on_time_bonus[tier]
                    slack_norm = slack_at_sched / max(job.runtime, 1e-3)
                    if slack_norm < 6.0:
                        reward += rw.tight_slack_on_time_bonus * max(0.0, 1.0 - slack_norm / 6.0)
                elif tier == 2:
                    reward -= rw.high_priority_miss_extra * miss_f
                elif tier == 1:
                    reward -= rw.mid_priority_miss_extra * miss_f
            self.t = finish_t
        else:
            self.t += 1.0
            if rw.use_wa3c:
                reward = self._wa3c_reward_idle()
            else:
                reward = -0.05

        dismissed = self._fill_arrivals()
        if rw.use_wa3c:
            reward += rw.w_dismiss * (-rw.wa3c_mu_dismissal * float(dismissed))
        else:
            q_frac = float(len(self.queue)) / float(max(1, self.max_queue))
            reward -= rw.queue_pressure_coef * (q_frac**2)
            if self.completed > 0:
                miss_rate = float(self.deadline_misses) / float(self.completed)
                reward -= rw.miss_rate_stability_coef * (miss_rate**2)

        terminated = self.completed >= len(self.jobs) or (self.idx >= len(self.jobs) and len(self.queue) == 0)
        truncated = self.step_count >= self.max_steps

        info = {
            "time": self.t,
            "completed": self.completed,
            "deadline_misses": self.deadline_misses,
            "mean_wait": self.total_wait / max(1, self.completed),
            "mean_energy": self.total_energy / max(1, self.completed),
            "dismissals_total": self.dismissals_total,
        }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
