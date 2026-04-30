#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep colors consistent with the publication palette in other scripts.
MODEL_COLORS = {
    "RL-MOTS": "#4D4D4D",
    "HDRL": "#0072B2",
    "PPO": "#56B4E9",
    "A3C": "#009E73",
    "DQN": "#E69F00",
    "AdaptiveSched-Base": "#D55E00",
    "AdaptiveSched-Hybrid": "#CC79A7",
}


def _plot_single_metric(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylab: str,
    out_base: Path,
) -> None:
    plt.figure(figsize=(9.4, 5.2))
    plt.style.use("seaborn-v0_8-whitegrid")

    algos = list(df["Algorithm"].astype(str).tolist())
    ys = df[metric_col].astype(float).to_numpy()
    x = np.arange(1)  # single group: "Algorithms"
    w = 0.85 / max(1, len(algos))

    for i, (a, y) in enumerate(zip(algos, ys)):
        plt.bar(
            x + (i - (len(algos) - 1) / 2) * w,
            [float(y)],
            width=w,
            color=MODEL_COLORS.get(a, "#666666"),
            label=a,
        )

    plt.xticks(x, ["Algorithms"])
    plt.ylabel(ylab)
    plt.title(title)

    # Put legend centered at bottom, outside axes (prevents overlap).
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    plt.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))

    plt.savefig(out_base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.savefig(out_base.with_suffix(".pdf"), dpi=400, bbox_inches="tight")
    plt.savefig(out_base.with_suffix(".svg"), dpi=400, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--table5_csv",
        type=str,
        default="benchmarks/outputs/s41598_final_paper_plus_ours/table5_comparable_performance_metrics.csv",
    )
    ap.add_argument("--out_dir", type=str, default="figures/s41598_final")
    args = ap.parse_args()

    df = pd.read_csv(args.table5_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_single_metric(df, "Completion Rate (%)", "Fig. 14. Completion rate", "Completion rate (%)", out_dir / "fig14_completion_rates")
    _plot_single_metric(df, "Energy Saving (%)", "Fig. 15. Energy saving", "Energy saving (%)", out_dir / "fig15_energy_saving")
    _plot_single_metric(df, "Cost Reduction (%)", "Fig. 16. Cost reduction", "Cost reduction (%)", out_dir / "fig16_cost_reduction")


if __name__ == "__main__":
    main()

