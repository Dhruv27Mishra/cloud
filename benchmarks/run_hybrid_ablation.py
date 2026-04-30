#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

FULL_ALGOS = [
    "ppo",
    "a2c",
    "sac_discrete",
    "dqn",
    "decision_transformer",
    "iql_discrete",
    "cql_discrete",
    "deeprm_pg",
    "decima_style",
    "deepjs_dqn",
    "performative_rl",
    "hybrid_role_based_marl",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="benchmarks/outputs/hybrid_ablation")
    ap.add_argument(
        "--scenarios",
        nargs="+",
        default=["000", "001", "002"],
        help="Scenario ids from datasets/synthetic/suite_v3_large",
    )
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--baseline_root",
        type=str,
        default="benchmarks/outputs/full_synthetic_v3_whole_suite_pub",
        help="Directory containing cached full-suite scenario_{id}/full_suite_comparison.csv files.",
    )
    ap.add_argument(
        "--algos",
        nargs="+",
        default=["hybrid_role_based_marl"],
        help="Algorithms to train in ablation runs (default: hybrid only; baselines are loaded from cached results).",
    )
    return ap.parse_args()


def _load_cached_rows(baseline_root: Path, scenario: str) -> List[Dict[str, str]]:
    p = baseline_root / f"scenario_{scenario}" / "full_suite_comparison.csv"
    if not p.exists():
        raise FileNotFoundError(f"Cached baseline file not found: {p}")
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_variant(
    root: Path,
    out_dir: Path,
    baseline_root: Path,
    scenario: str,
    variant_name: str,
    env_overrides: Dict[str, str],
    steps: int,
    seed: int,
    algos: List[str],
) -> Dict[str, str]:
    jobs_csv = f"datasets/synthetic/suite_v3_large/scenario_{scenario}/jobs.csv"
    run_out = out_dir / variant_name / f"scenario_{scenario}"
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["PROPOSED_FAST"] = "1"
    env.update(env_overrides)
    cmd = [
        "python3",
        "benchmarks/compare_full_suite.py",
        "--jobs_csv",
        jobs_csv,
        "--dataset_name",
        f"ablation_{variant_name}_s{scenario}",
        "--out_dir",
        str(run_out),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--algos",
        *algos,
    ]
    subprocess.run(cmd, cwd=root, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    p = run_out / "full_suite_comparison.csv"
    with p.open(newline="", encoding="utf-8") as f:
        new_rows = list(csv.DictReader(f))
    hyb = next(r for r in new_rows if r["algo"] == "hybrid_role_based_marl")

    cached = _load_cached_rows(baseline_root, scenario)
    merged = [r for r in cached if r["algo"] != "hybrid_role_based_marl"]
    merged.append(
        {
            "algo": "hybrid_role_based_marl",
            "model_path": hyb.get("model_path", ""),
            "steps": hyb.get("steps", ""),
            "mean_return": hyb["mean_return"],
            "mean_deadline_misses": hyb["mean_deadline_misses"],
            "mean_wait": hyb["mean_wait"],
            "mean_energy": hyb["mean_energy"],
            "composite_score": hyb["composite_score"],
        }
    )
    rows_sorted = sorted(merged, key=lambda r: float(r["composite_score"]), reverse=True)
    rank_map = {r["algo"]: i + 1 for i, r in enumerate(rows_sorted)}
    hyb_rank = rank_map["hybrid_role_based_marl"]
    top = rows_sorted[0]
    best_other = next((r for r in rows_sorted if r["algo"] != "hybrid_role_based_marl"), rows_sorted[0])
    n_competitors = max(0, len(rows_sorted) - 1)
    beats_all = sum(
        1
        for r in rows_sorted
        if r["algo"] != "hybrid_role_based_marl" and float(hyb["composite_score"]) > float(r["composite_score"])
    )
    return {
        "scenario": scenario,
        "variant": variant_name,
        "hybrid_rank": str(hyb_rank),
        "hybrid_composite": hyb["composite_score"],
        "hybrid_return": hyb["mean_return"],
        "hybrid_misses": hyb["mean_deadline_misses"],
        "hybrid_wait": hyb["mean_wait"],
        "top_algo": top["algo"],
        "top_composite": top["composite_score"],
        "best_other_algo": best_other["algo"],
        "best_other_composite": best_other["composite_score"],
        "beats_all_competitors_count": str(beats_all),
        "n_competitors": str(n_competitors),
        "wins_scenario": "yes" if hyb_rank == 1 else "no",
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline_root = Path(args.baseline_root)

    variants = {
        "full_hybrid": {
            "HYBRID_USE_ACTION_SHAPING": "1",
            "HYBRID_USE_Q_AUX": "1",
            "HYBRID_USE_Q_GUIDANCE": "0",
            "HYBRID_Q_COEF": "0.52",
            "HYBRID_CQL_ALPHA": "0.5",
            "PROPOSED_SHAPE_SCALE": "8.0",
            "HYBRID_COMPLETION_BONUS_COEF": "0.12",
            "COMP_W_MISS": "1.7",
            "COMP_W_WAIT": "0.08",
            "COMP_W_ENERGY": "0.05",
        },
        "no_action_shaping": {
            "HYBRID_USE_ACTION_SHAPING": "0",
            "HYBRID_USE_Q_AUX": "1",
            "HYBRID_USE_Q_GUIDANCE": "0",
            "HYBRID_Q_COEF": "0.52",
            "HYBRID_CQL_ALPHA": "0.5",
            "PROPOSED_SHAPE_SCALE": "8.0",
            "HYBRID_COMPLETION_BONUS_COEF": "0.12",
            "COMP_W_MISS": "1.7",
            "COMP_W_WAIT": "0.08",
            "COMP_W_ENERGY": "0.05",
        },
        "no_q_aux": {
            "HYBRID_USE_ACTION_SHAPING": "1",
            "HYBRID_USE_Q_AUX": "0",
            "HYBRID_USE_Q_GUIDANCE": "0",
            "PROPOSED_SHAPE_SCALE": "8.0",
            "HYBRID_COMPLETION_BONUS_COEF": "0.12",
            "COMP_W_MISS": "1.7",
            "COMP_W_WAIT": "0.08",
            "COMP_W_ENERGY": "0.05",
        },
        "no_q_guidance": {
            "HYBRID_USE_ACTION_SHAPING": "1",
            "HYBRID_USE_Q_AUX": "1",
            "HYBRID_USE_Q_GUIDANCE": "0",
            "HYBRID_Q_COEF": "0.52",
            "HYBRID_CQL_ALPHA": "0.5",
            "PROPOSED_SHAPE_SCALE": "8.0",
            "HYBRID_COMPLETION_BONUS_COEF": "0.12",
            "COMP_W_MISS": "1.7",
            "COMP_W_WAIT": "0.08",
            "COMP_W_ENERGY": "0.05",
        },
    }

    rows: List[Dict[str, str]] = []
    for scen in args.scenarios:
        for name, envv in variants.items():
            print(f"Run {name} scenario_{scen}", flush=True)
            rows.append(_run_variant(root, out, baseline_root, scen, name, envv, args.steps, args.seed, args.algos))

    detail = out / "ablation_detail.csv"
    with detail.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    agg = defaultdict(lambda: {"n": 0, "sum_comp": 0.0, "sum_rank": 0.0, "wins": 0, "beats_all": 0, "n_others": 0})
    for r in rows:
        a = agg[r["variant"]]
        a["n"] += 1
        a["sum_comp"] += float(r["hybrid_composite"])
        a["sum_rank"] += float(r["hybrid_rank"])
        a["wins"] += 1 if r["wins_scenario"] == "yes" else 0
        a["beats_all"] += int(r["beats_all_competitors_count"])
        a["n_others"] += int(r["n_competitors"])

    summary_rows = []
    for v, a in agg.items():
        summary_rows.append(
            {
                "variant": v,
                "n_scenarios": a["n"],
                "avg_hybrid_composite": f"{a['sum_comp']/a['n']:.6f}",
                "avg_hybrid_rank": f"{a['sum_rank']/a['n']:.4f}",
                "wins_count": a["wins"],
                "wins_rate": f"{a['wins']/a['n']:.4f}",
                "beats_all_total": a["beats_all"],
                "beats_all_rate": f"{a['beats_all']/max(1, a['n_others']):.4f}",
            }
        )
    summary_rows.sort(key=lambda x: float(x["avg_hybrid_rank"]))
    summary = out / "ablation_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Wrote {detail}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()

