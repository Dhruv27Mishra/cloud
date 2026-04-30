# Benchmark Models (Pre-Proposal Comparison Stack)

This module provides a model-comparison stack you asked for, to benchmark against your future proposed model.

## Implemented Baselines

- `ppo` (Stable-Baselines3 PPO)
- `a2c` (Stable-Baselines3 A2C)
- `sac_discrete` (discrete-action SAC implementation)
- `dqn` (**important extra baseline**) (Stable-Baselines3 DQN)

## Implemented Paper-Era Models (2018-2025 window)

These are practical implementations/adaptations for the scheduling benchmark:

1. `decision_transformer`  
   - Inspired by Decision Transformer (Chen et al., NeurIPS 2021).
2. `iql_discrete`  
   - Inspired by Implicit Q-Learning (Kostrikov et al., 2022).
3. `cql_discrete`  
   - Inspired by Conservative Q-Learning (Kumar et al., NeurIPS 2020).

> Note: some papers are not originally proposed for this exact discrete cloud-scheduling environment, so implementations are faithful adaptations for comparison.

## Cloud Scheduling Paper-Inspired Models (kept additionally)

- `deeprm_pg`  
  - DeepRM-style policy gradient scheduler (Mao et al., 2016 line).
- `decima_style`  
  - Decima-style learning scheduler (Mao et al., SIGCOMM 2019 inspired).
- `deepjs_dqn`  
  - DeepJS-style DQN scheduler (cloud DRL scheduling, 2019-era inspired).

## Proposed-model candidates (head-to-head)

Run `compare_proposed_models.py` to train and rank only:

- `performative_rl` — PPO-style updates with performative reward shaping.
- `hybrid_role_based_marl` — role-based policy heads + same performative shaping.

```bash
PYTHONPATH=.. python3 compare_proposed_models.py \
  --jobs_csv ../datasets/synthetic/suite_v1/scenario_000/jobs.csv \
  --out_dir outputs/proposed_two_way
```

---

## Dataset Schema Expected

Every dataset (real or synthetic) should provide a `jobs.csv` with columns:

- `job_id`
- `arrival_time`
- `runtime`
- `cpu_demand`
- `mem_demand`
- `priority`
- `deadline_time`

You can use synthetic suite outputs directly.

---

## Install

```bash
cd benchmarks
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run Training

```bash
python3 train_all.py \
  --jobs_csv ../datasets/synthetic/suite_v1/scenario_000/jobs.csv \
  --out_dir outputs/suite_v1_s000 \
  --steps 25000 \
  --seed 0 \
  --algos ppo a2c sac_discrete dqn decision_transformer iql_discrete cql_discrete deeprm_pg decima_style deepjs_dqn
```

Outputs:
- `outputs/.../models/*`
- `outputs/.../training_summary.csv`

---

## Real Datasets

Real dataset acquisition and manifests are under:
- `../datasets/real/`
- `../docs/datasets/README_DATASET_SUITE.md`

Because of provider policies and very large sizes, some traces require manual credentialed steps.
