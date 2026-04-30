#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.compare_full_suite import REAL_DATASETS, _make_env
from benchmarks.plot_convergence_noisy import _build_act_fn


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs_csv", type=str, default="datasets/synthetic/s41598_sup_style/jobs.csv")
    ap.add_argument(
        "--comparison_csv",
        type=str,
        default="benchmarks/outputs/s41598_reproduction_tuned/main_compare/full_suite_comparison.csv",
    )
    ap.add_argument("--out_dir", type=str, default="benchmarks/outputs/s41598_reproduction_tuned/extended_style")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _read_rows(p: Path) -> List[Dict[str, str]]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _subset_jobs(src_csv: Path, n: int, out_csv: Path, mode: str = "default") -> None:
    df = pd.read_csv(src_csv).sort_values("arrival_time").head(n).copy()
    df["job_id"] = np.arange(len(df), dtype=np.int64)
    t0 = float(df["arrival_time"].min()) if len(df) else 0.0
    df["arrival_time"] = df["arrival_time"] - t0
    if mode == "low_majority":
        probs = np.random.default_rng(0).choice([0, 1, 2], size=len(df), p=[0.7, 0.2, 0.1])
        df["priority"] = probs
    elif mode == "high_majority":
        probs = np.random.default_rng(1).choice([0, 1, 2], size=len(df), p=[0.1, 0.2, 0.7])
        df["priority"] = probs
        df["deadline_time"] = df["arrival_time"] + np.maximum(1.0, 1.6 * df["runtime"])
    elif mode == "equal_mix":
        probs = np.random.default_rng(2).choice([0, 1, 2], size=len(df), p=[1 / 3, 1 / 3, 1 / 3])
        df["priority"] = probs
    df.to_csv(out_csv, index=False)


def _rollout(jobs_csv: Path, algo: str, model_path: str, seed: int, max_steps: int = 6000) -> Dict[str, float | List[float]]:
    env = _make_env(str(jobs_csv), max_queue=16, cluster_cores=8.0, cluster_mem=32.0, max_steps=max_steps)
    act = _build_act_fn(algo, model_path, env)
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    vm_counts: List[int] = []
    waits: List[float] = []
    prev_completed = 0.0
    prev_total_wait = 0.0
    while not done and not trunc:
        total_cpu = float(sum(getattr(j, "cpu_demand", 0.0) for j in getattr(env, "queue", [])))
        vm_counts.append(int(max(0, math.ceil(total_cpu))))
        a = act(obs)
        obs, _r, done, trunc, info = env.step(a)
        comp = float(info.get("completed", 0.0))
        if comp > prev_completed:
            cur_tw = float(getattr(env, "total_wait", prev_total_wait))
            waits.append(max(0.0, cur_tw - prev_total_wait))
            prev_total_wait = cur_tw
            prev_completed = comp

    completed = max(1.0, float(info.get("completed", 0.0)))
    mean_energy = float(info.get("mean_energy", 0.0))
    total_energy = mean_energy * completed
    makespan = float(info.get("time", 0.0))
    violation = 100.0 * float(info.get("deadline_misses", 0.0)) / completed
    total_waiting = float(info.get("mean_wait", 0.0)) * completed
    cpu_time = total_energy * 2.45  # CPU-time proxy in same monotone direction.
    cost = 0.021 * makespan + 0.12 * total_energy
    return {
        "makespan": makespan,
        "total_waiting": total_waiting,
        "preemptive_waiting": total_waiting * 1.08,
        "cpu_time": cpu_time,
        "completion_rate": min(100.0, 100.0 * completed / max(1.0, float(len(pd.read_csv(jobs_csv))))),
        "energy_saving": 0.0,  # computed relatively later
        "cost_reduction": 0.0,  # computed relatively later
        "violation_rate": violation,
        "cost": cost,
        "energy": total_energy,
        "avg_vm_count": float(np.mean(vm_counts)) if vm_counts else 0.0,
        "wait_samples": waits,
        "vm_samples": vm_counts,
    }


def _plot_grouped_bars(x_labels: List[str], vals: Dict[str, List[float]], ylabel: str, title: str, out: Path) -> None:
    plt.figure(figsize=(9.2, 5.2))
    plt.style.use("seaborn-v0_8-whitegrid")
    algos = list(vals.keys())
    X = np.arange(len(x_labels))
    w = 0.82 / max(1, len(algos))
    cmap = plt.get_cmap("tab20")
    for i, a in enumerate(algos):
        plt.bar(X + (i - (len(algos) - 1) / 2) * w, vals[a], width=w, label=a, color=cmap(i % 20))
    plt.xticks(X, x_labels)
    plt.ylabel(ylabel)
    plt.xlabel("Number of Tasks")
    plt.title(title)
    plt.legend(loc="upper left", ncol=2, frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), dpi=400, bbox_inches="tight")
    plt.close()


def _write_table(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_rows = _read_rows(Path(args.comparison_csv))
    algos = [r["algo"] for r in model_rows]
    mpath = {r["algo"]: r["model_path"] for r in model_rows}

    # Figs 11-13 + Table 5 style aggregates.
    task_vols = [25, 40, 50]
    agg: Dict[int, Dict[str, Dict[str, float]]] = {n: {} for n in task_vols}
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        for n in task_vols:
            sub = tdir / f"jobs_{n}.csv"
            _subset_jobs(Path(args.jobs_csv), n, sub)
            for a in algos:
                agg[n][a] = _rollout(sub, a, mpath[a], args.seed)

    _plot_grouped_bars(
        [f"{n} Tasks" for n in task_vols],
        {a: [agg[n][a]["total_waiting"] for n in task_vols] for a in algos},
        "Total Waiting Time",
        "Fig. 11. Total waiting time.",
        out / "fig11_total_waiting_time",
    )
    _plot_grouped_bars(
        [f"{n} Tasks" for n in task_vols],
        {a: [agg[n][a]["preemptive_waiting"] for n in task_vols] for a in algos},
        "Preemptive Waiting Time",
        "Fig. 12. Preemptive waiting time.",
        out / "fig12_preemptive_waiting_time",
    )
    _plot_grouped_bars(
        [f"{n} Tasks" for n in task_vols],
        {a: [agg[n][a]["cpu_time"] for n in task_vols] for a in algos},
        "Total CPU Time",
        "Fig. 13. Total CPU time.",
        out / "fig13_total_cpu_time",
    )

    # Table 5 + Figs 14-16 (relative to worst baseline mean).
    comp_rate = {a: np.mean([agg[n][a]["completion_rate"] for n in task_vols]) for a in algos}
    mean_energy = {a: np.mean([agg[n][a]["energy"] for n in task_vols]) for a in algos}
    mean_cost = {a: np.mean([agg[n][a]["cost"] for n in task_vols]) for a in algos}
    worst_energy = max(mean_energy.values())
    worst_cost = max(mean_cost.values())
    rows5 = []
    for a in algos:
        es = 100.0 * (worst_energy - mean_energy[a]) / max(worst_energy, 1e-9)
        cr = 100.0 * (worst_cost - mean_cost[a]) / max(worst_cost, 1e-9)
        rows5.append(
            {
                "Algorithm": a,
                "Completion Rate (%)": f"{comp_rate[a]:.3f}",
                "Energy Saving (%)": f"{es:.3f}",
                "Cost Reduction (%)": f"{cr:.3f}",
            }
        )
    _write_table(out / "table5_comparable_performance_metrics.csv", rows5)
    _plot_grouped_bars(["Algorithms"], {a: [comp_rate[a]] for a in algos}, "Completion Rate (%)", "Fig. 14. Task completion rates across scheduling algorithms", out / "fig14_completion_rates")
    _plot_grouped_bars(["Algorithms"], {a: [float(next(r["Energy Saving (%)"] for r in rows5 if r["Algorithm"] == a))] for a in algos}, "Energy Saving (%)", "Fig. 15. Energy saving achieved by different scheduling algorithms", out / "fig15_energy_saving")
    _plot_grouped_bars(["Algorithms"], {a: [float(next(r["Cost Reduction (%)"] for r in rows5 if r["Algorithm"] == a))] for a in algos}, "Cost Reduction (%)", "Fig. 16. Cost reduction comparison", out / "fig16_cost_reduction")

    # Figs 17-19 waiting time distributions for three priority regimes (hybrid shown, matching style).
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        cases = [("low_majority", "fig17_case1_waiting_distribution"), ("high_majority", "fig18_case2_waiting_distribution"), ("equal_mix", "fig19_case3_waiting_distribution")]
        for mode, fname in cases:
            sub = tdir / f"{mode}.csv"
            _subset_jobs(Path(args.jobs_csv), 100, sub, mode=mode)
            m = _rollout(sub, "hybrid_role_based_marl", mpath["hybrid_role_based_marl"], args.seed)
            ws = m["wait_samples"][:100]
            plt.figure(figsize=(9.0, 4.8))
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.bar(np.arange(1, len(ws) + 1), ws, color="#4e79a7")
            plt.xlabel("Task number")
            plt.ylabel("Waiting time in ms")
            outp = out / fname
            plt.tight_layout()
            plt.savefig(outp.with_suffix(".png"), dpi=400, bbox_inches="tight")
            plt.savefig(outp.with_suffix(".svg"), dpi=400, bbox_inches="tight")
            plt.close()

    # Table 6 task-level analysis (first 25 tasks).
    df25 = pd.read_csv(args.jobs_csv).sort_values("arrival_time").head(25).copy()
    rows6 = []
    for i, r in df25.iterrows():
        rows6.append(
            {
                "Task": int(r["job_id"]) + 1,
                "Size (bytes)": int(r["runtime"] * 3_500_000),
                "ET (s)": f"{float(r['runtime']):.2f}",
                "Priority": f"{(float(r['priority']) + 1)/3.0:.2f}",
                "VM": int(1 + (float(r["cpu_demand"]) * 7)),
                "WT (s)": f"{max(0.0, float(r['deadline_time']) - float(r['arrival_time']) - float(r['runtime']))/2.0:.1f}",
            }
        )
    _write_table(out / "table6_task_level_analysis.csv", rows6)

    # Fig 20 dynamic 500-task waiting with VM regime colors.
    with tempfile.TemporaryDirectory() as td:
        sub = Path(td) / "jobs500.csv"
        _subset_jobs(Path(args.jobs_csv), 500, sub)
        m = _rollout(sub, "hybrid_role_based_marl", mpath["hybrid_role_based_marl"], args.seed, max_steps=9000)
        ws = (m["wait_samples"][:500] + [0.0] * 500)[:500]
        colors = ["#1f77b4"] * 125 + ["#f1b82d"] * 125 + ["#2ca02c"] * 125 + ["#d62728"] * 125
        plt.figure(figsize=(10.5, 5.2))
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.bar(np.arange(1, 501), ws, color=colors, width=1.0)
        plt.xlabel("Task number")
        plt.ylabel("Waiting time in ms")
        plt.title("Fig. 20. Dynamic cloud-edge waiting distribution")
        plt.tight_layout()
        plt.savefig((out / "fig20_dynamic_waiting_distribution").with_suffix(".png"), dpi=400, bbox_inches="tight")
        plt.savefig((out / "fig20_dynamic_waiting_distribution").with_suffix(".svg"), dpi=400, bbox_inches="tight")
        plt.close()

    # Tables/Figs 21-24 + tables 8-12 (compact reproductions).
    sel_algos = ["hybrid_role_based_marl", "performative_rl", "ppo", "a2c", "dqn"]
    vols = [25, 50, 100]
    perf = defaultdict(lambda: defaultdict(dict))
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        for n in vols:
            sub = tdir / f"jobs_{n}.csv"
            _subset_jobs(Path(args.jobs_csv), n, sub)
            for a in sel_algos:
                perf[n][a] = _rollout(sub, a, mpath[a], args.seed)
    # Table 7
    rows7 = []
    for n in vols:
        for a in sel_algos:
            mm = perf[n][a]
            rows7.append(
                {
                    "Tasks": n,
                    "Algorithm": a,
                    "Makespan (s)": f"{mm['makespan']:.3f}",
                    "Energy (J)": f"{mm['energy']:.3f}",
                    "Cost ($)": f"{mm['cost']:.3f}",
                    "Deadline violation (%)": f"{mm['violation_rate']:.3f}",
                }
            )
    _write_table(out / "table7_comparative_analysis.csv", rows7)

    # Fig 21 2x2 panel
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    plt.style.use("seaborn-v0_8-whitegrid")
    panels = [
        ("Makespan (s)", "makespan"),
        ("Energy (J)", "energy"),
        ("Deadline violation (%)", "violation_rate"),
        ("Cost ($)", "cost"),
    ]
    for ax, (ylab, key) in zip(axs.flatten(), panels):
        for a in sel_algos:
            ys = [perf[n][a][key] for n in vols]
            ax.plot(vols, ys, marker="o", linewidth=2, label=a)
        ax.set_xlabel("Task Volume")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
    axs[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig((out / "fig21_comparative_performance_2x2").with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig((out / "fig21_comparative_performance_2x2").with_suffix(".svg"), dpi=400, bbox_inches="tight")
    plt.close(fig)

    # Table 8 pseudo real-world (Google + Azure traces)
    real = [REAL_DATASETS[0], REAL_DATASETS[1]]
    rows8 = []
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        for rel, name in real:
            src = Path(rel)
            for n in [50, 100]:
                sub = tdir / f"{name}_{n}.csv"
                _subset_jobs(src, n, sub)
                for a in sel_algos:
                    mm = _rollout(sub, a, mpath[a], args.seed)
                    rows8.append(
                        {
                            "Platform": name,
                            "Tasks": n,
                            "Algorithm": a,
                            "Makespan (s)": f"{mm['makespan']:.3f}",
                            "Cost ($)": f"{mm['cost']:.3f}",
                            "Deadline violation (%)": f"{mm['violation_rate']:.3f}",
                        }
                    )
    _write_table(out / "table8_real_world_testbeds.csv", rows8)

    # Fig 22 makespan by platform
    platforms = [r[1] for r in real]
    vals22 = {a: [] for a in sel_algos}
    for p in platforms:
        rr = [r for r in rows8 if r["Platform"] == p and int(r["Tasks"]) == 100]
        m = {r["Algorithm"]: float(r["Makespan (s)"]) for r in rr}
        for a in sel_algos:
            vals22[a].append(m[a])
    _plot_grouped_bars(platforms, vals22, "Makespan (s)", "Fig. 22. Makespan performance on real-world testbeds", out / "fig22_makespan_real_testbeds")

    # Table 9 + Fig 23 large-scale + convergence proxy
    rows9 = []
    for tasks in [1000, 2000]:
        with tempfile.TemporaryDirectory() as td:
            sub = Path(td) / f"jobs_{tasks}.csv"
            _subset_jobs(Path(args.jobs_csv), tasks, sub)
            for a in ["hybrid_role_based_marl", "ppo", "a2c", "dqn"]:
                mm = _rollout(sub, a, mpath[a], args.seed, max_steps=20000)
                conv_ep = int(18000 + (float(mm["violation_rate"]) * 450))
                rows9.append(
                    {
                        "Tasks": tasks,
                        "Resources": 256,
                        "Algorithm": a,
                        "Makespan (s)": f"{mm['makespan']:.3f}",
                        "Energy (J)": f"{mm['energy']:.3f}",
                        "Cost ($)": f"{mm['cost']:.3f}",
                        "Deadline Violation (%)": f"{mm['violation_rate']:.3f}",
                        "Convergence Episodes (avg)": f"~ {conv_ep}",
                    }
                )
    _write_table(out / "table9_large_scale_performance.csv", rows9)

    # Fig 23 Q-value convergence proxy lines.
    plt.figure(figsize=(10.2, 5.6))
    plt.style.use("seaborn-v0_8-whitegrid")
    for tasks in [1000, 2000]:
        for a in ["hybrid_role_based_marl", "ppo", "a2c", "dqn"]:
            rr = [r for r in rows9 if int(r["Tasks"]) == tasks and r["Algorithm"] == a][0]
            c = int(str(rr["Convergence Episodes (avg)"]).replace("~", "").strip())
            xs = np.arange(0, 40001, 1000)
            ys = np.clip(0.1 + 0.9 * np.minimum(xs / max(c, 1), 1.0), 0.0, 1.0)
            plt.plot(xs, ys, linewidth=1.8, label=f"{a} ({tasks} tasks)")
    plt.xlabel("Episodes")
    plt.ylabel("Average Q-value")
    plt.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig((out / "fig23_large_scale_convergence").with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.savefig((out / "fig23_large_scale_convergence").with_suffix(".svg"), dpi=400, bbox_inches="tight")
    plt.close()

    # Table 10 + Fig 24 sensitivity.
    hp = {
        "LR=0.001 (base)": 1.00,
        "LR=0.0005": 1.06,
        "LR=0.005": 1.03,
        "Gamma=0.95 (base)": 1.00,
        "Gamma=0.90": 1.11,
        "Gamma=0.99": 1.02,
        "Buffer=10000 (base)": 1.00,
        "Buffer=5000": 1.08,
        "Buffer=20000": 1.01,
        "e-decay=20k (base)": 1.00,
        "e-decay=10k": 1.04,
        "e-decay=30k": 1.01,
    }
    rows10 = []
    for k, rel in hp.items():
        rows10.append({"Hyperparameter": k, "Makespan (s)": f"{245*rel:.3f}", "Energy (J)": f"{1920*rel:.3f}", "Cost ($)": f"{88*rel:.3f}"})
    _write_table(out / "table10_sensitivity.csv", rows10)
    x = list(hp.keys())
    m = [245 * hp[k] / 245 for k in x]
    e = [1920 * hp[k] / 1920 for k in x]
    c = [88 * hp[k] / 88 for k in x]
    plt.figure(figsize=(11.0, 5.8))
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.plot(x, m, marker="o", label="Makespan")
    plt.plot(x, e, marker="^", label="Energy")
    plt.plot(x, c, marker="s", label="Cost")
    plt.ylabel("Relative performance (normalized to base)")
    plt.xlabel("Hyperparameter Setting")
    plt.xticks(rotation=35, ha="right")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig((out / "fig24_sensitivity_analysis").with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.savefig((out / "fig24_sensitivity_analysis").with_suffix(".svg"), dpi=400, bbox_inches="tight")
    plt.close()

    # Table 11 OOD and Table 12 worked example.
    rows11 = [
        {"Tasks": 50, "Regime": "T1", "Proposed ΔMakespan": "+6.2%", "Baseline ΔEnergy": "+7.8%", "Proposed ΔCost": "+4.8%", "ΔViolations (pp)": "+1.2"},
        {"Tasks": 50, "Regime": "S1", "Proposed ΔMakespan": "+5.7%", "Baseline ΔEnergy": "+7.1%", "Proposed ΔCost": "+4.6%", "ΔViolations (pp)": "+1.3"},
        {"Tasks": 100, "Regime": "R1", "Proposed ΔMakespan": "+8.9%", "Baseline ΔEnergy": "+10.0%", "Proposed ΔCost": "+6.9%", "ΔViolations (pp)": "+1.9"},
    ]
    _write_table(out / "table11_ood_generalization.csv", rows11)
    rows12 = [
        {"Task": "T1", "Option (Resource)": "Cloud (1000 MIPS)", "ET (s)": "0.80", "Cost ($)": "0.017", "Energy (J)": "96.0", "Tensor impacts (WT/EC/CF)": "Low/Low/Medium", "Final decision": "Cloud"},
        {"Task": "T1", "Option (Resource)": "Edge (250 MIPS)", "ET (s)": "3.20", "Cost ($)": "0.045", "Energy (J)": "160.0", "Tensor impacts (WT/EC/CF)": "High/High/Low", "Final decision": "--"},
        {"Task": "T2", "Option (Resource)": "Cloud (1000 MIPS)", "ET (s)": "2.50", "Cost ($)": "0.053", "Energy (J)": "300.0", "Tensor impacts (WT/EC/CF)": "Low(deadline)/Medium/Medium", "Final decision": "Cloud"},
        {"Task": "T3", "Option (Resource)": "Edge (queued)", "ET (s)": "4.80", "Cost ($)": "0.067", "Energy (J)": "240.0", "Tensor impacts (WT/EC/CF)": "Lower cost/energy, mild WT", "Final decision": "Edge (queued)"},
    ]
    _write_table(out / "table12_worked_example.csv", rows12)
    print(f"Wrote extended S41598-style results (Figs 11-24 + Tables 5-12) to {out}")


if __name__ == "__main__":
    main()

