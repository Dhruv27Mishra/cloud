# Synthetic Suite Specification

Use (Chen et al.-style synthetic workload preset enabled by default):

```bash
python3 scripts/data_suite/generate_synthetic_suite.py \
  --out_dir datasets/synthetic/suite_v1 \
  --n_scenarios 18 \
  --jobs_per_scenario 60000 \
  --seed 42 \
  --chen_style 1
```

Large-scale variant used for reviewer concerns on scale:

```bash
python3 scripts/data_suite/generate_synthetic_suite.py \
  --out_dir datasets/synthetic/suite_v3_large \
  --n_scenarios 8 \
  --jobs_per_scenario 120000 \
  --seed 42 \
  --chen_style 1
```

Generated per scenario:
- `jobs.csv`
- `scenario_config.json`

Generated at suite root:
- `suite_index.csv`

Core properties:
- non-stationary arrivals (trend + seasonality + bursts),
- Markov-modulated low/high traffic states (burst persistence),
- mutable priority mix with optional shock-time priority inversion,
- Chen-style job-class mixture (`short`, `medium`, `long`) with class-specific runtimes,
- correlated CPU/MEM demands with heavy-tail resource jobs,
- priority-coupled deadline strictness (high-priority tighter deadlines),
- shock windows and pressure feedback.
