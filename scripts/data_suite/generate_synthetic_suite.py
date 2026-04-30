#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ScenarioConfig:
    scenario_id: str
    seed: int
    jobs: int
    horizon_seconds: int
    base_arrival_rate: float
    trend_strength: float
    season_amp: float
    season_period: int
    burst_factor: float
    pressure_feedback: float
    prio_low: float
    prio_med: float
    prio_high: float
    runtime_logn_mu: float
    runtime_logn_sigma: float
    cpu_mem_corr: float
    deadline_multiplier: float
    deadline_noise: float
    shock_start: int
    shock_duration: int
    shock_arrival_mult: float
    shock_priority_flip: int
    # Chen-style workload controls (A2C/A3C scheduling literature)
    chen_style: int
    low_state_mult: float
    high_state_mult: float
    p_stay_low: float
    p_stay_high: float
    short_job_share: float
    medium_job_share: float
    long_job_share: float
    short_logn_mu: float
    short_logn_sigma: float
    medium_logn_mu: float
    medium_logn_sigma: float
    long_logn_mu: float
    long_logn_sigma: float
    heavy_cpu_prob: float
    heavy_mem_prob: float
    deadline_mult_hi: float
    deadline_mult_md: float
    deadline_mult_lo: float


def _poisson_knuth(rng: random.Random, lam: float) -> int:
    if lam <= 0.0:
        return 0
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= rng.random()
    return k - 1


def _pick_priority(rng: random.Random, low: float, med: float, high: float) -> int:
    x = rng.random()
    if x < low:
        return 0
    if x < low + med:
        return 1
    return 2


def _bounded(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sample_scenario_config(idx: int, seed: int, jobs: int) -> ScenarioConfig:
    rng = random.Random(seed + 1000 * idx)
    horizon = rng.randint(6 * 3600, 48 * 3600)
    pr_l = _bounded(rng.uniform(0.45, 0.75), 0.05, 0.9)
    pr_m = _bounded(rng.uniform(0.15, 0.4), 0.05, 0.8)
    pr_h = _bounded(1.0 - pr_l - pr_m, 0.05, 0.5)
    s = pr_l + pr_m + pr_h
    pr_l, pr_m, pr_h = pr_l / s, pr_m / s, pr_h / s
    shock_start = rng.randint(int(0.2 * horizon), int(0.75 * horizon))
    shock_dur = rng.randint(600, 3600)
    short_share = _bounded(rng.uniform(0.45, 0.7), 0.2, 0.8)
    med_share = _bounded(rng.uniform(0.2, 0.4), 0.05, 0.6)
    long_share = _bounded(1.0 - short_share - med_share, 0.05, 0.35)
    s_cls = short_share + med_share + long_share
    short_share, med_share, long_share = short_share / s_cls, med_share / s_cls, long_share / s_cls
    return ScenarioConfig(
        scenario_id=f"scenario_{idx:03d}",
        seed=seed + idx,
        jobs=jobs,
        horizon_seconds=horizon,
        base_arrival_rate=rng.uniform(2.0, 15.0),
        trend_strength=rng.uniform(-0.8, 1.2),
        season_amp=rng.uniform(0.0, 1.2),
        season_period=rng.choice([300, 600, 900, 1800]),
        burst_factor=rng.uniform(1.0, 3.5),
        pressure_feedback=rng.uniform(0.0, 1.6),
        prio_low=pr_l,
        prio_med=pr_m,
        prio_high=pr_h,
        runtime_logn_mu=rng.uniform(1.5, 4.2),
        runtime_logn_sigma=rng.uniform(0.3, 1.1),
        cpu_mem_corr=rng.uniform(0.0, 1.0),
        deadline_multiplier=rng.uniform(1.2, 6.0),
        deadline_noise=rng.uniform(0.05, 0.6),
        shock_start=shock_start,
        shock_duration=shock_dur,
        shock_arrival_mult=rng.uniform(1.0, 3.2),
        shock_priority_flip=rng.choice([0, 1]),
        chen_style=1,
        low_state_mult=rng.uniform(0.55, 0.9),
        high_state_mult=rng.uniform(1.3, 2.2),
        p_stay_low=rng.uniform(0.92, 0.985),
        p_stay_high=rng.uniform(0.9, 0.98),
        short_job_share=short_share,
        medium_job_share=med_share,
        long_job_share=long_share,
        short_logn_mu=rng.uniform(0.2, 1.3),
        short_logn_sigma=rng.uniform(0.18, 0.42),
        medium_logn_mu=rng.uniform(1.6, 2.5),
        medium_logn_sigma=rng.uniform(0.25, 0.58),
        long_logn_mu=rng.uniform(2.8, 4.0),
        long_logn_sigma=rng.uniform(0.32, 0.85),
        heavy_cpu_prob=rng.uniform(0.18, 0.42),
        heavy_mem_prob=rng.uniform(0.15, 0.36),
        deadline_mult_hi=rng.uniform(1.05, 1.75),
        deadline_mult_md=rng.uniform(1.6, 2.8),
        deadline_mult_lo=rng.uniform(2.4, 4.5),
    )


def _arrival_rate(cfg: ScenarioConfig, t: int, pressure_state: float, state_mult: float) -> float:
    x = t / max(1.0, cfg.horizon_seconds)
    trend = 1.0 + cfg.trend_strength * (x - 0.5)
    seasonal = 1.0 + cfg.season_amp * math.sin(2.0 * math.pi * t / max(1, cfg.season_period))
    # daily/diurnal phase (synthetic clock)
    diurnal = 1.0 + 0.35 * math.sin(2.0 * math.pi * t / max(1, 3600))
    burst = cfg.burst_factor if random.random() < 0.0025 else 1.0
    shock = 1.0
    if cfg.shock_start <= t < cfg.shock_start + cfg.shock_duration:
        shock = cfg.shock_arrival_mult
    feedback = 1.0 + cfg.pressure_feedback * _bounded(pressure_state, 0.0, 1.5)
    lam = (
        cfg.base_arrival_rate
        * max(0.05, trend)
        * max(0.05, seasonal)
        * max(0.05, diurnal)
        * max(0.05, state_mult)
        * burst
        * shock
        * feedback
    )
    return _bounded(lam, 0.05, 120.0)


def _sample_runtime(rng: random.Random, cfg: ScenarioConfig) -> Tuple[float, str]:
    x = rng.random()
    if x < cfg.short_job_share:
        return max(0.15, rng.lognormvariate(cfg.short_logn_mu, cfg.short_logn_sigma)), "short"
    if x < cfg.short_job_share + cfg.medium_job_share:
        return max(0.4, rng.lognormvariate(cfg.medium_logn_mu, cfg.medium_logn_sigma)), "medium"
    return max(1.0, rng.lognormvariate(cfg.long_logn_mu, cfg.long_logn_sigma)), "long"


def _resource_pair(rng: random.Random, cfg: ScenarioConfig, cls: str) -> Tuple[float, float]:
    # Correlated CPU/MEM with occasional heavy tails, as seen in trace-driven cluster studies.
    if cls == "short":
        c0 = rng.uniform(0.05, 0.45)
    elif cls == "medium":
        c0 = rng.uniform(0.12, 0.72)
    else:
        c0 = rng.uniform(0.25, 1.0)
    if rng.random() < cfg.heavy_cpu_prob:
        c0 = _bounded(c0 + rng.uniform(0.2, 0.45), 0.05, 1.0)
    mem = cfg.cpu_mem_corr * c0 + (1.0 - cfg.cpu_mem_corr) * rng.uniform(0.05, 1.0) + rng.uniform(-0.12, 0.12)
    if rng.random() < cfg.heavy_mem_prob:
        mem += rng.uniform(0.15, 0.4)
    return _bounded(c0, 0.05, 1.0), _bounded(mem, 0.05, 1.0)


def _deadline_multiplier_by_priority(cfg: ScenarioConfig, prio: int) -> float:
    if prio >= 2:
        return cfg.deadline_mult_hi
    if prio == 1:
        return cfg.deadline_mult_md
    return cfg.deadline_mult_lo


def _gen_jobs(cfg: ScenarioConfig) -> List[Dict[str, float]]:
    rng = random.Random(cfg.seed)
    jobs: List[Dict[str, float]] = []
    pressure = 0.0
    job_id = 0
    t = 0
    # Two-state Markov arrival process (low/high load) to mimic burst phases.
    arr_state = 0
    while len(jobs) < cfg.jobs and t < cfg.horizon_seconds:
        if cfg.chen_style:
            if arr_state == 0:
                arr_state = 0 if rng.random() < cfg.p_stay_low else 1
            else:
                arr_state = 1 if rng.random() < cfg.p_stay_high else 0
        state_mult = cfg.low_state_mult if arr_state == 0 else cfg.high_state_mult
        lam = _arrival_rate(cfg, t, pressure, state_mult)
        arrivals = _poisson_knuth(rng, lam)
        for _ in range(arrivals):
            if len(jobs) >= cfg.jobs:
                break
            if cfg.chen_style:
                runtime, cls = _sample_runtime(rng, cfg)
                base_cpu, mem = _resource_pair(rng, cfg, cls)
            else:
                runtime = max(1.0, rng.lognormvariate(cfg.runtime_logn_mu, cfg.runtime_logn_sigma))
                cls = "mixed"
                base_cpu = _bounded(rng.random() ** 0.55, 0.05, 1.0)
                mem_noise = rng.uniform(-0.2, 0.2)
                mem = _bounded(cfg.cpu_mem_corr * base_cpu + (1.0 - cfg.cpu_mem_corr) * rng.random() + mem_noise, 0.05, 1.0)
            prio = _pick_priority(rng, cfg.prio_low, cfg.prio_med, cfg.prio_high)
            if cfg.shock_priority_flip and cfg.shock_start <= t < cfg.shock_start + cfg.shock_duration:
                prio = 2 - prio
            ddl_mul = _deadline_multiplier_by_priority(cfg, prio) if cfg.chen_style else cfg.deadline_multiplier
            ddl = t + runtime * ddl_mul * (1.0 + rng.uniform(-cfg.deadline_noise, cfg.deadline_noise))
            jobs.append(
                {
                    "job_id": job_id,
                    "arrival_time": t,
                    "runtime": round(runtime, 6),
                    "cpu_demand": round(base_cpu, 6),
                    "mem_demand": round(mem, 6),
                    "priority": prio,
                    "deadline_time": round(max(t + 1.0, ddl), 6),
                    "scenario_id": cfg.scenario_id,
                    "job_class": cls,
                }
            )
            job_id += 1
        # Pressure proxy: if arrival is above base, pressure rises; else decays.
        pressure = _bounded(0.96 * pressure + 0.04 * max(0.0, (lam / max(0.1, cfg.base_arrival_rate)) - 1.0), 0.0, 2.5)
        t += 1
    return jobs


def _write_jobs(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "job_id",
        "arrival_time",
        "runtime",
        "cpu_demand",
        "mem_demand",
        "priority",
        "deadline_time",
        "scenario_id",
        "job_class",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_scenarios", type=int, default=12)
    parser.add_argument("--jobs_per_scenario", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chen_style",
        type=int,
        default=1,
        help="1 enables Chen-style synthetic workload preset (default).",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    suite_rows: List[Dict[str, str]] = []
    for i in range(args.n_scenarios):
        cfg = _sample_scenario_config(i, args.seed, args.jobs_per_scenario)
        cfg.chen_style = int(args.chen_style)
        scen_dir = out / cfg.scenario_id
        scen_dir.mkdir(parents=True, exist_ok=True)

        rows = _gen_jobs(cfg)
        _write_jobs(scen_dir / "jobs.csv", rows)
        (scen_dir / "scenario_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        suite_rows.append(
            {
                "scenario_id": cfg.scenario_id,
                "seed": str(cfg.seed),
                "jobs": str(len(rows)),
                "horizon_seconds": str(cfg.horizon_seconds),
                "base_arrival_rate": f"{cfg.base_arrival_rate:.6f}",
                "trend_strength": f"{cfg.trend_strength:.6f}",
                "season_amp": f"{cfg.season_amp:.6f}",
                "burst_factor": f"{cfg.burst_factor:.6f}",
                "pressure_feedback": f"{cfg.pressure_feedback:.6f}",
                "deadline_multiplier": f"{cfg.deadline_multiplier:.6f}",
                "shock_arrival_mult": f"{cfg.shock_arrival_mult:.6f}",
                "chen_style": str(cfg.chen_style),
                "short_job_share": f"{cfg.short_job_share:.6f}",
                "medium_job_share": f"{cfg.medium_job_share:.6f}",
                "long_job_share": f"{cfg.long_job_share:.6f}",
                "low_state_mult": f"{cfg.low_state_mult:.6f}",
                "high_state_mult": f"{cfg.high_state_mult:.6f}",
            }
        )
        print(f"Generated {cfg.scenario_id}: {len(rows)} jobs")

    index_path = out / "suite_index.csv"
    fields = list(suite_rows[0].keys()) if suite_rows else []
    with index_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(suite_rows)
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
