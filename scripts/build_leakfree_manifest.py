"""Restrict the STEAD manifest to the official STEAD test split.

The split comes from the EQTransformer repository (``test.npy``, 126,566 trace
names) and is also the split used by SeisBench's STEAD benchmark.  Pretrained
PhaseNet-stead / EQTransformer weights were trained on the complementary
training traces, so this subset is a leak-free check for them.

Usage::

    python scripts/build_leakfree_manifest.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/smousavi05/EQTransformer/"
    "master/ModelsAndSampleData/test.npy"
)
OFFICIAL = Path("data/stead_official_test.npy")
MANIFEST = Path("data/stead_manifest.csv")
OUT = Path("data/stead_manifest_leakfree.csv")


def ensure_official() -> np.ndarray:
    if not OFFICIAL.exists():
        print(f"downloading official STEAD test split -> {OFFICIAL}")
        OFFICIAL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, OFFICIAL)
    return np.load(OFFICIAL, allow_pickle=True)


def main() -> None:
    official = set(ensure_official().tolist())
    frame = pd.read_csv(MANIFEST)
    subset = frame[frame["trace_name"].isin(official)].copy()
    subset.to_csv(OUT, index=False)
    print(f"official test traces: {len(official):,}")
    print(subset.groupby(["split", "category"]).size())
    print(f"wrote {OUT} ({len(subset):,} rows)")


if __name__ == "__main__":
    main()
