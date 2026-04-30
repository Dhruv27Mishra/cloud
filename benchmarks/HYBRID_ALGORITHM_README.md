# Hybrid Role-Based MARL: Algorithmic Design Notes (Publication Draft)

This document explains the implemented `hybrid_role_based_marl` method in `benchmarks/models/proposed_models.py` at an algorithmic and RL level, including novelty vs borrowed components and rationale.

## 1) Problem Setup

We solve discrete queue-slot scheduling in `TraceSchedulingEnv`:

- state `s_t`: flattened queue slot features + global counters,
- action `a_t`: select one queue index,
- transition: execute selected job if valid, otherwise idle step,
- reward: WA3C-style weighted multi-objective scalarization (QoS / energy / priority / fairness / dismissal) in `TraceSchedulingEnv`, aligned with A3C/cluster-scheduling reward-design traditions and multi-objective scalarization practice [Mnih2016, Mao2016, Chen2018].

The method is optimized for high-load, non-stationary multi-priority workloads.

## 2) Hybrid Policy Architecture

`RoleBasedPolicyValue` contains:

- shared encoder `f_theta(s)` (MLP + LayerNorm),
- three role-specific policy heads:
  - low-priority head,
  - medium-priority head,
  - high-priority head,
- value head `V(s)`,
- conservative double-Q heads `Q1(s,·), Q2(s,·)` (hybrid-only auxiliary heads) [Fujimoto2018, Kumar2020].

Role mixture weights are extracted from queue composition:

- compute current queue class histogram over low/med/high priority slots,
- sharpen with exponent `role_mix_sharp`,
- normalized vector `w_role`,
- final policy logits:
  `pi_logits = w_lo * head_lo + w_md * head_md + w_hi * head_hi`.

This makes role activation state-adaptive rather than fixed.

## 3) Action Selection: Learned Policy + Reward-Aligned Shaping

We do not rely on raw policy logits alone. We build shaped logits:

1. **Policy logits** from role-mixed actor.
2. **One-step heuristic score** (reward-aligned) approximating immediate utility:
   - QoS term from latency proxy,
   - priority satisfaction term,
   - energy penalty proxy,
   - fairness-gain proxy (variance reduction when serving a slot),
   - urgency term from negative slack,
   - hopelessness penalty for severely infeasible jobs,
   - runtime cost regularization.
3. **Priority-weighted softmax bias** term (`beta * priority_norm`).
4. **Adaptive blending** between learned logits and shaped logits based on queue pressure and urgency.

This is the practical bridge between model-based scheduling intuition and policy learning [Mao2016, Chen2018].

## 4) Training Objective (Hybrid)

Hybrid uses PPO backbone plus auxiliary conservative value learning:

- **PPO clipped policy objective** (on shaped-action distribution) [Schulman2017],
- **value regression** via `V(s)` and GAE returns [Schulman2016],
- **entropy regularization**,
- **auxiliary CQL-style double-Q loss**:
  - TD target with clipped-double-Q bootstrap (`max_a min(Q1,Q2)`) [Fujimoto2018],
  - conservative term `logsumexp(Q) - Q(a_data)` [Kumar2020],
- **advantage-weighted behavior cloning regularizer** to stabilize updates on high-advantage actions [Nair2020, Peng2019].

Overall:

`L = L_PPO + c_v * L_V - c_ent * H + c_q * L_QTD + c_cql * L_CQL + c_bc * L_AWBC`.

Additionally, in role mode we add a small completion-rate bonus to dense reward during rollout collection to favor actionable progress under congestion.

## 5) What Is Novel vs Borrowed

### Borrowed / standard components

- PPO clipped surrogate + GAE + value/entropy terms [Schulman2017, Schulman2016],
- conservative Q-learning regularizer form (`logsumexp - Q_data`) [Kumar2020],
- double-Q anti-overestimation intuition [Fujimoto2018, Hasselt2010],
- advantage-weighted cloning as a stabilizer [Nair2020, Peng2019].

### Proposed integration novelty

1. **Role-conditioned policy mixture over queue composition**.
2. **Reward-aligned action-logit shaping integrated directly into policy distribution**.
3. **Hybrid actor-critic + conservative double-Q auxiliary heads in one unified scheduler**.
4. **Pressure-adaptive blending / coefficient behavior for high-load regions**.
5. **Priority/urgency/hopelessness-aware slot scoring combined with learned policy**.

The novelty claim is therefore in the *combination and scheduler-specific coupling* rather than inventing PPO/CQL from scratch.

## 6) Why These Choices

- Pure policy-gradient methods can be unstable under bursty workload shifts [Mnih2016, Schulman2017].
- Pure conservative value methods can be robust but less reactive to role structure [Kumar2020].
- Role-head decomposition improves interpretability and specialization across priority classes.
- Logit shaping injects domain priors where sparse RL credit assignment is weak.
- Conservative auxiliary Q discourages overconfident scheduling choices under distribution drift.

Together, this yields better behavior under overloaded and non-stationary queues, a regime emphasized in prior RL scheduling systems [Mao2016, Chen2018].

## 7) Ablation Protocol (Implemented)

The implementation supports hybrid ablations via env variables:

- `HYBRID_USE_ACTION_SHAPING` (`1/0`)
- `HYBRID_USE_Q_AUX` (`1/0`)
- `HYBRID_USE_Q_GUIDANCE` (`1/0`, default `0` after ablation)
- `HYBRID_ROLE_MIX_SHARP`
- `HYBRID_Q_COEF`, `HYBRID_CQL_ALPHA`, `HYBRID_BC_COEF`
- `PROPOSED_SHAPE_SCALE`
- `HYBRID_COMPLETION_BONUS_COEF`
- `PROPOSED_FAST=1` (fast tuning mode)

This enables publication-grade component attribution and reproducibility.

## 8) Reproducibility Notes

- Main training/eval entrypoint: `benchmarks/compare_full_suite.py`
- Proposed models: `benchmarks/models/proposed_models.py`
- Synthetic suites: `datasets/synthetic/suite_v2_chen`, `datasets/synthetic/suite_v3_large`
- Use fixed seeds and report both aggregate rank and metric-level tradeoffs (misses/wait/energy/return).

## 9) References

- [Mnih2016] Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. ICML.
- [Schulman2016] Schulman, J., et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR.
- [Schulman2017] Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- [Mao2016] Mao, H., Alizadeh, M., Menache, I., and Kandula, S. (2016). Resource Management with Deep Reinforcement Learning. HotNets.
- [Chen2018] Chen, X., et al. (2018). Efficient Multi-Resource Allocation in Cloud Data Centers via Deep Reinforcement Learning. (A2C/A3C-style cloud scheduling line of work).
- [Fujimoto2018] Fujimoto, S., van Hoof, H., and Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. ICML.
- [Kumar2020] Kumar, A., et al. (2020). Conservative Q-Learning for Offline Reinforcement Learning. NeurIPS.
- [Hasselt2010] van Hasselt, H. (2010). Double Q-learning. NeurIPS.
- [Nair2020] Nair, A., et al. (2020). Accelerating Online Reinforcement Learning with Offline Datasets. arXiv:2006.09359.
- [Peng2019] Peng, X. B., et al. (2019). Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning. arXiv:1910.00177.

