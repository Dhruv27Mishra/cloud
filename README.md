# AdaptiveSched Research Artifact

Code and reproducibility assets for trace-driven cloud job scheduling experiments.

This repository intentionally excludes manuscript source/PDF bundles and large raw datasets.
Use the instructions below to recreate datasets and rerun experiments.

## Repository Layout

- `benchmarks/` - model implementations, training scripts, evaluators, and plotting pipelines
- `scripts/data_suite/` - dataset download/processing/generation utilities
- `figures/s41598_final/` - final figure assets used by the reported results
- `docs/` - methodology and dataset notes

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r benchmarks/requirements.txt
```

## Dataset Preparation

Create datasets locally (they are not committed to git):

### 1) Download real-trace inputs

```bash
bash scripts/data_suite/download_real_datasets.sh
```

This script prepares local directories and fetches publicly accessible artifacts. Some sources (for example Alibaba) require manual gated download steps described in script output.

### 2) Build normalized job tables

```bash
python scripts/data_suite/prepare_google_task_events_jobs.py
python scripts/data_suite/prepare_azure_vmtable_jobs.py
python scripts/data_suite/prepare_alibaba_batch_jobs.py
```

### 3) Generate synthetic suites

```bash
python scripts/data_suite/generate_synthetic_suite.py
python scripts/data_suite/generate_s41598_sup_style_dataset.py
```

## Reproduce Main Results

Repository reference: https://github.com/Dhruv27Mishra/cloud

From repository root:

```bash
cd benchmarks
python make_s41598_paper_plus_ours_validated.py
python run_hybrid_ablation.py
python regenerate_single_panel_results.py
```

Outputs are written under `benchmarks/outputs/`.

## Notes

- Large raw traces are excluded to keep the repository GitHub-compatible.
- Use `docs/datasets/README_DATASET_SUITE.md` and script-level help for source-specific details.
