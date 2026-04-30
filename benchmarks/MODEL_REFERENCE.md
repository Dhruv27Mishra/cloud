# Model Reference (for comparison suite)

## Core baselines

- PPO (Schulman et al., 2017)
- A2C / actor-critic family (Mnih et al., 2016 lineage)
- SAC (Haarnoja et al., 2018; discrete adaptation used here)
- DQN baseline (Mnih et al., 2015) as an important value-based reference for discrete scheduling

## Paper-era models implemented (2018-2025 relevant window)

1. Decision Transformer (Chen et al., NeurIPS 2021) -> `decision_transformer`
2. Implicit Q-Learning (Kostrikov et al., 2022) -> `iql_discrete`
3. Conservative Q-Learning (Kumar et al., NeurIPS 2020) -> `cql_discrete`

These are implemented as practical adaptations to the discrete cloud scheduling setting.

## Cloud scheduling application models (additional)

1. DeepRM-style scheduler (Mao et al., 2016 line) -> `deeprm_pg`
2. Decima-style scheduler (Mao et al., SIGCOMM 2019 line) -> `decima_style`
3. DeepJS-style DQN scheduler (cloud DRL scheduling line, 2019-era) -> `deepjs_dqn`

Note: these are practical reproductions/inspired implementations for unified comparison in this benchmark environment.

## Proposed-model sweep (two candidates)

- `performative_rl` — `benchmarks/models/proposed_models.py`
- `hybrid_role_based_marl` — role-conditioned logits + performative term; same module.

Use `benchmarks/compare_proposed_models.py` and `benchmarks/run_publication_sweep_proposed.py`.
