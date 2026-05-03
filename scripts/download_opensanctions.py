"""
Download OpenSanctions ownership data for the 50% rule graph analysis.

Uses the OpenSanctions FTM (FollowTheMoney) bulk format — each line of the
.ftm.json file is a JSON object representing one entity or relationship.

Datasets downloaded:
    ru_rupep        — Russian politically exposed persons with ownership
                      relationships from Transparency International Russia.
                      Richest source for post-2022 ownership graph data.

    us_ofac_sdn     — OFAC SDN list in FTM format (for cross-referencing
                      entity IDs with our parsed Advanced XML data).

Output: data/raw/os_ru_rupep.ftm.json
        data/raw/os_us_ofac_sdn.ftm.json

Run from the project root:
    uv run --env-file .env python scripts/download_opensanctions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://data.opensanctions.org/datasets/latest"

DATASETS = {
    "os_ru_rupep.ftm.json":    f"{BASE}/ru_rupep/entities.ftm.json",
    "os_us_ofac_sdn.ftm.json": f"{BASE}/us_ofac_sdn/entities.ftm.json",
}


def download(url: str, dest: Path, chunk_mb: int = 4) -> None:
    print(f"  → {dest.name} ... ", end="", flush=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
            f.write(chunk)
            total += len(chunk)
    print(f"{total / 1e6:.1f} MB")


def main() -> None:
    for fname, url in DATASETS.items():
        dest = RAW / fname
        if dest.exists():
            print(f"  Skipping {fname} (already downloaded)")
            continue
        download(url, dest)
    print("\nDone.")


if __name__ == "__main__":
    main()
