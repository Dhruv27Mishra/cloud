# Dataset Suite for Cloud Scheduling RL

This workspace now includes a practical dataset suite with:

1. **Real datasets** used in cloud scheduling / systems papers.
2. A **controlled synthetic suite** with configurable and mutable workload shifts.

The goal is to benchmark multiple scheduling models under both realistic traces and stress-tested synthetic scenarios.

---

## Included Real Datasets (Paper-Relevant)

The downloader targets these commonly used families:

- **Google Cluster Data 2011 (Borg trace)**
  - Source: [google/cluster-data](https://github.com/google/cluster-data)
  - Notes: Data is hosted in a Google Storage bucket (`clusterdata-2011-2`), usually fetched with `gcloud storage`.
- **Alibaba Cluster Trace 2018**
  - Source: [alibaba/clusterdata](https://github.com/alibaba/clusterdata)
  - Notes: Full trace download requires survey / gated links.
- **Azure Public Dataset V2 (2019 VM traces)**
  - Source: [Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset)
  - Notes: Direct blob links are published in a text file.
- **Bitbrains traces (VM workload traces often used in scheduling/placement literature)**
  - Source metadata is scaffolded; download endpoint can be filled in if your preferred mirror is used.

---

## Directory Layout

- `datasets/real/`
  - `google_cluster_2011/`
  - `alibaba_cluster_2018/`
  - `azure_vm_2019/`
  - `bitbrains/`
- `datasets/synthetic/`
  - generated benchmark suites (CSV + metadata JSON)
- `scripts/data_suite/`
  - `download_real_datasets.py`
  - `generate_synthetic_suite.py`

---

## 1) Download Real Datasets

### Quick start (metadata + manifests only)

```bash
python3 scripts/data_suite/download_real_datasets.py --mode metadata
```

This creates dataset cards, source references, and machine-readable manifests under `datasets/real/**`.

### Attempt automated downloads where possible

```bash
python3 scripts/data_suite/download_real_datasets.py --mode full
```

What this does:
- Pulls public docs/manifests from GitHub.
- Pulls Azure link manifest (`AzurePublicDatasetLinksV2.txt`).
- Writes step-by-step commands for Google (`gcloud storage`) and Alibaba (survey/fetch script).
- Attempts direct HTTP downloads only when safe and explicit URLs are known.

> Full raw traces are very large (tens to hundreds of GB). The script is designed to be resumable and explicit rather than silently failing.

---

## 2) Generate Controlled + Mutable Synthetic Suite

Generate a diverse suite:

```bash
python3 scripts/data_suite/generate_synthetic_suite.py \
  --out_dir datasets/synthetic/suite_v1 \
  --n_scenarios 18 \
  --jobs_per_scenario 60000 \
  --seed 42
```

Each scenario includes:
- `jobs.csv`: tabular jobs with arrival, runtime, cpu, mem, priority, deadline.
- `scenario_config.json`: exact knobs used.

And the root includes:
- `suite_index.csv`: summary of all generated scenarios.

### Mutation knobs implemented

- Arrival process: base rate + trend + seasonal pattern + burst factor.
- Performative-like feedback proxy: load-pressure induced arrival multiplier.
- Priority mix shift: low/med/high class probabilities.
- Runtime shape drift: lognormal distribution shifts.
- Resource correlation: CPU-MEM coupling strength.
- Deadline strictness and deadline noise.
- Shock events: temporary load spikes and priority inversions.
- Non-stationarity windows: piecewise parameter changes across time segments.

This allows you to test:
- robustness,
- distribution shift tolerance,
- adaptation behavior,
- ablation sensitivity.

---

## Suggested Benchmark Protocol

For each model:
1. Train on:
   - Google 2011 subset,
   - Alibaba 2018 subset,
   - synthetic "base" scenarios.
2. Evaluate on:
   - held-out real trace slices,
   - synthetic shifted scenarios (bursty / strict deadlines / high-correlation / shock).
3. Report:
   - reward,
   - latency (mean + P95),
   - deadline miss rate,
   - drop/dismissal rate,
   - energy proxy,
   - fairness.

---

## Notes

- Some real datasets require terms acceptance or gated links (especially Alibaba full trace).
- Keep raw archives outside Git and version only processed metadata/splits.
- Prefer deterministic scenario generation with fixed seeds for reproducibility.
