# Proposed Algorithms Architecture and Working

This document explains the two proposed scheduling algorithms:

- `performative_rl`
- `hybrid_role_based_marl`

It is written in two styles:

- **Technical view** for implementation and research details
- **Layman view** for intuitive understanding

It also lists:

- what is **novel** in this work
- what is **borrowed/adapted** from prior RL ideas
- pseudocode for reference

---

## 1) Problem Context

We schedule jobs arriving over time in a shared compute cluster.
Each job has:

- runtime
- CPU demand
- memory demand
- priority
- deadline

At each step, the agent picks one queued job slot to run.  
Good scheduling should balance:

- low deadline misses (SLA)
- low waiting time
- good throughput/completion
- reasonable energy and cost
- fair resource usage

---

## 2) High-Level Difference Between the Two Proposed Algorithms

- **`performative_rl`**: one strong actor-critic policy learns a global scheduling strategy.
- **`hybrid_role_based_marl`**: role-aware multi-head policy where different heads specialize for low/medium/high priority job dynamics, then blend their decisions.

Think of `performative_rl` as one expert scheduler, and `hybrid_role_based_marl` as a team of specialists coordinated together.

---

## 3) Technical Architecture

## 3.1 `performative_rl` (single-policy actor-critic)

### Network

- Shared MLP backbone for policy/value estimation
- Policy head outputs logits over queue slots
- Value head estimates state value for PPO-style updates

### Action Selection

1. Encode observation
2. Compute policy logits
3. Apply reward-aligned action shaping (priority/urgency/fairness-aware)
4. Sample (train) or argmax (eval) valid slot action

### Learning

- On-policy PPO-like objective:
  - clipped policy loss
  - value regression loss
  - entropy regularization
- GAE for advantage estimation
- Optional performative regularization term (distribution-shift robustness flavor)

### When It Works Best

- scenarios where one coherent policy can generalize across queue states
- lower architectural complexity than hybrid

---

## 3.2 `hybrid_role_based_marl` (role-conditioned hybrid actor-critic)

### Network

- Shared encoder over full state
- Three role-specific policy heads:
  - low-priority specialist
  - medium-priority specialist
  - high-priority specialist
- Role-mixture weights inferred from current queue composition
- Value head for policy learning
- Auxiliary double-Q heads (`q1`, `q2`) for conservative/stable guidance terms

### Action Selection

1. Extract role weights from occupied queue slots
2. Compute each role-head logits
3. Weighted blend of role-head logits
4. Apply reward-aligned shaping bonus (urgency, hopelessness, wait pressure, fairness gain)
5. Select action

### Learning

- Same PPO core loop as above, plus optional hybrid auxiliaries:
  - double-Q TD guidance
  - conservative (CQL-inspired) regularization
  - advantage-weighted behavior-cloning regularizer
- Pressure-aware scaling of auxiliary penalties
- Role mixture sharpening for specialization

### Why It Helps

- Queue contains heterogeneous job regimes; separate role heads specialize and reduce policy interference.
- Better control over SLA-critical decisions under priority/deadline imbalance.

---

## 4) End-to-End Flow (Both Algorithms)

1. Build environment from workload trace (`jobs.csv`)
2. Reset env and observe queue-state vector
3. Policy outputs action preference over queue slots
4. Action shaping adjusts logits toward SLA-aligned behavior
5. Environment executes selected job, returns reward and metrics
6. Collect rollout trajectories
7. Compute advantages/returns
8. Update policy/value (and hybrid auxiliaries if enabled)
9. Repeat until step budget ends
10. Save checkpoint and evaluate on metrics:
    - return
    - deadline misses
    - waiting time
    - energy
    - fairness/throughput (in extended analyses)

---

## 5) Layman-Friendly Explanation

Imagine a busy hospital triage desk:

- Patients arrive with different urgency, complexity, and waiting limits.
- You must choose who gets treatment next.

### `performative_rl`

- One experienced triage nurse learns from past outcomes.
- They become good at balancing urgency, fairness, and efficiency.

### `hybrid_role_based_marl`

- Instead of one nurse, you have 3 specialists:
  - specialist for low urgency
  - specialist for medium urgency
  - specialist for high urgency
- A coordinator looks at the current waiting room mix and decides how much to trust each specialist.
- This often improves decisions when the queue has mixed urgency patterns.

In short: hybrid is a coordinated specialist team; performative is a strong generalist.

---

## 6) Novelty vs Borrowed Components

## 6.1 Novel Contributions in This Work

- **Role-conditioned policy blending for scheduling**:
  queue-driven dynamic mixing of low/medium/high role heads.
- **Reward-aligned action shaping**:
  blends learned logits with urgency/priority/fairness-aware heuristic signals.
- **Pressure-adaptive hybrid guidance**:
  auxiliary conservative/value terms scale with queue pressure.
- **SLA-focused integration**:
  architecture and shaping jointly tuned toward deadline-sensitive scheduling goals.

## 6.2 Borrowed / Adapted Foundations

- **PPO-style actor-critic training** (policy/value/entropy structure)
- **GAE** for advantage estimation
- **Double-Q idea** for reduced value overestimation
- **CQL-inspired conservative regularization** (adapted to this discrete scheduling context)
- **Behavior-cloning style regularizer** (advantage-weighted variant)
- Standard MLP actor/critic parameterization for discrete actions

This is not claiming PPO/CQL/Double-Q as new; the novelty is in **how they are combined and adapted** to this role-aware, SLA-driven scheduling system.

---

## 7) Practical Interpretation of Metrics

- If **deadline violation drops**, SLA behavior improves.
- If **makespan drops**, workload finishes sooner.
- If **cost and energy drop**, efficiency improves.
- If **fairness rises**, resource allocation becomes less skewed.

The hybrid model is intended to dominate in mixed-priority, high-pressure queues, while performative remains a strong and simpler baseline.

---

## 8) Pseudocode (Reference)

```text
Algorithm 1: Performative RL Scheduling
Input: env E, policy/value net θ, total steps T
for step = 1 ... T do
    observe state s_t
    logits, V(s_t) <- Netθ(s_t)
    logits' <- ActionShape(E, s_t, logits)    # priority/urgency/fairness aligned
    a_t ~ Categorical(logits')
    s_{t+1}, r_t <- E.step(a_t)
    r_t <- r_t - λ_perf * ShiftPenalty(E)     # optional performative term
    store transition
    if rollout complete then
        compute GAE advantages and returns
        update θ with PPO clipped objective + value loss + entropy bonus
    end if
end for
return trained model
```

```text
Algorithm 2: Hybrid Role-Based MARL Scheduling
Input: env E, shared encoder + role heads + value + optional Q heads, total steps T
for step = 1 ... T do
    observe state s_t
    w_role <- ExtractRoleWeights(s_t queue composition)
    logits_role, V(s_t) <- RoleNet(s_t, w_role)
    logits' <- ActionShape(E, s_t, logits_role) + RoleBonus(s_t)
    a_t ~ Categorical(logits')
    s_{t+1}, r_t <- E.step(a_t)
    r_t <- r_t + CompletionBonus - λ_perf * ShiftPenalty(E)
    store transition
    if rollout complete then
        compute GAE advantages and returns
        loss <- PPO(policy,value,entropy)
        if Q-aux enabled then
            add DoubleQ TD loss
            add CQL-inspired conservative penalty
            add advantage-weighted BC regularizer
        end if
        update parameters
    end if
end for
return trained hybrid model
```

---

## 9) Quick Summary

- `performative_rl` = simpler, strong generalist actor-critic with reward-aligned shaping.
- `hybrid_role_based_marl` = role-specialist ensemble with adaptive blending and optional conservative value guidance.
- Novelty is in **role-aware coordination + SLA-aligned shaping + pressure-adaptive hybridization** for cloud job scheduling.
