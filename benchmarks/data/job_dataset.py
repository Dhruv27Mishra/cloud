from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


REQUIRED_COLUMNS = {
    "job_id",
    "arrival_time",
    "runtime",
    "cpu_demand",
    "mem_demand",
    "priority",
    "deadline_time",
}


@dataclass
class Job:
    job_id: int
    arrival_time: float
    runtime: float
    cpu_demand: float
    mem_demand: float
    priority: int
    deadline_time: float


def load_jobs_csv(path: str | Path) -> List[Job]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {p}")
    df = pd.read_csv(p)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {p}: {sorted(missing)}")
    df = df.sort_values("arrival_time").reset_index(drop=True)
    jobs: List[Job] = []
    for _, r in df.iterrows():
        jobs.append(
            Job(
                job_id=int(r["job_id"]),
                arrival_time=float(r["arrival_time"]),
                runtime=max(1e-6, float(r["runtime"])),
                cpu_demand=float(r["cpu_demand"]),
                mem_demand=float(r["mem_demand"]),
                priority=int(r["priority"]),
                deadline_time=float(r["deadline_time"]),
            )
        )
    return jobs
