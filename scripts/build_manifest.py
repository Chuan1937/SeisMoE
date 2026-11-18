"""Join the HDF5 offset index with STEAD labels and build event-level splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CSV = "/mnt/f/STEAD/STEAD.csv"
OUT = "data/stead_manifest.csv"


def main() -> None:
    names = np.load("data/stead_names.npy", allow_pickle=True)
    offsets = np.load("data/stead_offsets.npy")
    print(f"names={len(names):,} offsets={len(offsets):,}")

    index = pd.DataFrame({"trace_name": names, "offset": offsets})
    df = pd.read_csv(CSV, low_memory=False)
    df = df[
        [
            "trace_name",
            "trace_category",
            "p_arrival_sample",
            "s_arrival_sample",
            "source_id",
            "receiver_code",
        ]
    ]
    df["trace_name"] = df["trace_name"].astype(str)
    merged = index.merge(df, on="trace_name", how="inner")
    print(f"joined {len(merged):,} labelled traces")

    merged["p_sample"] = pd.to_numeric(merged["p_arrival_sample"], errors="coerce")
    merged["s_sample"] = pd.to_numeric(merged["s_arrival_sample"], errors="coerce")
    is_event = merged["trace_category"].astype(str).str.startswith("earthquake")
    valid = (
        is_event
        & merged["p_sample"].notna()
        & merged["s_sample"].notna()
        & (merged["p_sample"] > 0)
        & (merged["s_sample"] > merged["p_sample"])
        & (merged["s_sample"] < 6000)
    )
    event = merged[valid].copy()
    noise = merged[~is_event].copy()
    print(f"events={len(event):,} noise={len(noise):,}")

    # ---- event-level split by source_id -----------------------------------
    sources = np.array(sorted(event["source_id"].astype(str).unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(sources)
    n_test = int(round(len(sources) * 0.10))
    n_val = int(round(len(sources) * 0.10))
    test_sources = set(sources[:n_test])
    val_sources = set(sources[n_test : n_test + n_val])
    event["split"] = "train"
    event.loc[event["source_id"].astype(str).isin(val_sources), "split"] = "val"
    event.loc[event["source_id"].astype(str).isin(test_sources), "split"] = "test"

    # Cap val/test (complete sources, sampled traces).
    def cap(split: str, limit: int) -> pd.DataFrame:
        part = event[event["split"] == split]
        if len(part) > limit:
            part = part.sample(n=limit, random_state=42)
        return part

    event = pd.concat(
        [event[event["split"] == "train"], cap("val", 15000), cap("test", 25000)],
        ignore_index=True,
    )

    # ---- noise split --------------------------------------------------------
    import hashlib

    def noise_split(name: str) -> str:
        value = int(hashlib.md5(("noise:" + name).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "test" if value < 0.10 else ("val" if value < 0.20 else "train")

    noise = noise.copy()
    noise["split"] = noise["trace_name"].map(noise_split)
    noise = pd.concat(
        [
            noise[noise["split"] == "train"],
            noise[noise["split"] == "val"].head(4000),
            noise[noise["split"] == "test"].head(6000),
        ],
        ignore_index=True,
    )

    out = pd.concat([event, noise], ignore_index=True)
    out["category"] = np.where(
        out["trace_category"].astype(str).str.startswith("earthquake"), "event", "noise"
    )
    out["p_sample"] = out["p_sample"].fillna(-1).astype(np.int64)
    out["s_sample"] = out["s_sample"].fillna(-1).astype(np.int64)
    out["offset"] = out["offset"].astype(np.int64)
    cols = [
        "trace_name",
        "offset",
        "category",
        "split",
        "p_sample",
        "s_sample",
        "source_id",
        "receiver_code",
    ]
    out = out[cols].sort_values(["split", "trace_name"], ignore_index=True)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.groupby(["split", "category"]).size())
    print(f"wrote {OUT} ({len(out):,} rows)")
    with open("data/stead_split.json", "w") as handle:
        json.dump(
            {
                "sources": int(len(sources)),
                "val_sources": n_val,
                "test_sources": n_test,
                "rows": {
                    f"{k[0]}_{k[1]}": int(v)
                    for k, v in out.groupby(["split", "category"]).size().items()
                },
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
