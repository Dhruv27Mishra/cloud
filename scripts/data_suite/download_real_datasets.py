#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import textwrap
import urllib.request
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
REAL_ROOT = ROOT / "datasets" / "real"


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _download(url: str, dst: Path, timeout: int = 60) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        dst.write_bytes(data)
        return True
    except Exception:
        return False


def _which(cmd: str) -> bool:
    return subprocess.call(["bash", "-lc", f"command -v {cmd} >/dev/null 2>&1"]) == 0


def google_cluster(mode: str) -> Dict[str, str]:
    out = REAL_ROOT / "google_cluster_2011"
    _mkdir(out)
    card = textwrap.dedent(
        """\
        name: Google Cluster Data 2011 (v2.1)
        source_repo: https://github.com/google/cluster-data
        trace_bucket: gs://clusterdata-2011-2
        size_compressed: ~41GB
        notes:
          - Usually downloaded with gcloud storage tools.
          - Join mailing list with reason field for documentation updates.
        """
    )
    _write_text(out / "dataset_card.txt", card)

    _download(
        "https://raw.githubusercontent.com/google/cluster-data/master/ClusterData2011_2.md",
        out / "ClusterData2011_2.md",
    )

    instructions = textwrap.dedent(
        """\
        # Google Cluster 2011 download instructions
        # Requires Google Cloud SDK.
        # Example commands:
        #   gcloud storage ls gs://clusterdata-2011-2/
        #   gcloud storage cp --recursive gs://clusterdata-2011-2 ./raw
        #
        # Optional: pull only task_events first for scheduling experiments:
        #   gcloud storage ls gs://clusterdata-2011-2/task_events/
        #   gcloud storage cp --recursive gs://clusterdata-2011-2/task_events ./raw/task_events
        """
    )
    _write_text(out / "DOWNLOAD_INSTRUCTIONS.txt", instructions)

    if mode == "full" and _which("gcloud"):
        raw = out / "raw"
        _mkdir(raw)
        # Keep automated sync lightweight by listing only; user can run full copy manually.
        subprocess.call(
            [
                "bash",
                "-lc",
                f"gcloud storage ls gs://clusterdata-2011-2 > '{(out / 'bucket_listing.txt').as_posix()}'",
            ]
        )

    return {"dataset": "google_cluster_2011", "status": "prepared"}


def alibaba_cluster(mode: str) -> Dict[str, str]:
    out = REAL_ROOT / "alibaba_cluster_2018"
    _mkdir(out)
    _download(
        "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2018/trace_2018.md",
        out / "trace_2018.md",
    )
    _download(
        "https://raw.githubusercontent.com/alibaba/clusterdata/master/README.md",
        out / "alibaba_clusterdata_README.md",
    )

    instructions = textwrap.dedent(
        """\
        # Alibaba cluster-trace-v2018
        Canonical docs: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018

        Full data is typically distributed after a short survey:
          http://alibabadeveloper.mikecrm.com/BdJtacN

        In the upstream repository, fetch scripts and checksums are documented.
        After download, put archives under:
          datasets/real/alibaba_cluster_2018/raw_archives/

        Then extract into:
          datasets/real/alibaba_cluster_2018/raw/
        """
    )
    _write_text(out / "DOWNLOAD_INSTRUCTIONS.txt", instructions)
    _mkdir(out / "raw_archives")
    _mkdir(out / "raw")
    return {"dataset": "alibaba_cluster_2018", "status": "prepared_manual_download"}


def azure_vm(mode: str) -> Dict[str, str]:
    out = REAL_ROOT / "azure_vm_2019"
    _mkdir(out)
    ok1 = _download(
        "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetV2.md",
        out / "AzurePublicDatasetV2.md",
    )
    ok2 = _download(
        "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzurePublicDatasetLinksV2.txt",
        out / "AzurePublicDatasetLinksV2.txt",
    )
    _write_text(
        out / "dataset_card.txt",
        textwrap.dedent(
            """\
            name: Azure Public Dataset V2 (VM trace)
            source_repo: https://github.com/Azure/AzurePublicDataset
            compressed_size: ~156GB
            period: 30 days
            """
        ),
    )
    if mode == "full" and ok2:
        links = (out / "AzurePublicDatasetLinksV2.txt").read_text(encoding="utf-8").splitlines()
        links = [ln.strip() for ln in links if ln.strip().startswith("http")]
        dl_dir = out / "raw"
        _mkdir(dl_dir)
        # Download only first 2 files by default to avoid accidental 100GB+ transfer.
        sample = links[:2]
        for i, url in enumerate(sample):
            _download(url, dl_dir / f"sample_part_{i:03d}.csv.gz", timeout=180)
        _write_text(
            out / "FULL_DOWNLOAD_NOTE.txt",
            "Sample files downloaded. Edit script if you want complete 156GB transfer.",
        )

    return {"dataset": "azure_vm_2019", "status": "prepared" if ok1 else "partial"}


def bitbrains(mode: str) -> Dict[str, str]:
    out = REAL_ROOT / "bitbrains"
    _mkdir(out)
    _write_text(
        out / "dataset_card.txt",
        textwrap.dedent(
            """\
            name: Bitbrains VM workload traces
            notes:
              - Frequently referenced in VM placement/scheduling studies.
              - Mirrors vary by paper and institution.
            """
        ),
    )
    _write_text(
        out / "DOWNLOAD_INSTRUCTIONS.txt",
        textwrap.dedent(
            """\
            Provide your preferred Bitbrains mirror URL(s) in:
              datasets/real/bitbrains/bitbrains_urls.txt
            Then run this script again in --mode full to download them.
            """
        ),
    )
    urls_file = out / "bitbrains_urls.txt"
    if not urls_file.exists():
        _write_text(urls_file, "# one URL per line\n")
    if mode == "full":
        lines = [x.strip() for x in urls_file.read_text(encoding="utf-8").splitlines()]
        urls = [x for x in lines if x and not x.startswith("#")]
        raw = out / "raw"
        _mkdir(raw)
        for i, url in enumerate(urls):
            name = os.path.basename(url.split("?")[0]) or f"bitbrains_{i:03d}.dat"
            _download(url, raw / name, timeout=180)
    return {"dataset": "bitbrains", "status": "prepared"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["metadata", "full"], default="metadata")
    args = parser.parse_args()

    _mkdir(REAL_ROOT)
    summary: List[Dict[str, str]] = []
    summary.append(google_cluster(args.mode))
    summary.append(alibaba_cluster(args.mode))
    summary.append(azure_vm(args.mode))
    summary.append(bitbrains(args.mode))

    _write_text(REAL_ROOT / "download_summary.json", json.dumps(summary, indent=2))
    print(f"Wrote {REAL_ROOT / 'download_summary.json'}")


if __name__ == "__main__":
    main()
