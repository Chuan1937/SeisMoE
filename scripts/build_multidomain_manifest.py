"""Combine STEAD, TEXD, INSTANCE and DiTing into one byte-offset manifest.

Every row describes a trace that can be read with ``os.pread(path, nbytes,
offset)``:

* STEAD: contiguous ``(6000, 3)`` float32, ENZ, layout ``TC``.
* TEXD / INSTANCE: bucket datasets ``(1024, 3, T)``; item offset =
  bucket offset + index * T * 3 * itemsize, ZNE, layout ``CT``.
* DiTing 2.0: one contiguous ``(T, 3)`` int32 dataset per trace, ZNE,
  layout ``TC``.

``start`` is the first sample of the 60 s window (5 s before P); ``p_win`` /
``s_win`` are the labels inside that window.  No waveform is copied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

STEAD = {
    "path": "/mnt/f/STEAD/STEAD.hdf5",
    "samples": 6000,
    "nbytes": 6000 * 3 * 4,
    "dtype": "float32",
    "layout": "TC",
    "channel_order": "ENZ",
}
TEXD = {
    "path": "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
    "csv": "/mnt/d/AI_Seismic_Data/txed/Texd_filtered.csv",
    "p_col": "trace_p_arrival_sample",
    "s_col": "trace_s_arrival_sample",
    "samples": 6000,
    "dtype": "float64",
    "layout": "CT",
    "channel_order": "ZNE",
}
INSTANCE = {
    "path": "/mnt/d/AI_Seismic_Data/INSTANCE/INSTANCE.hdf5",
    "csv": "/mnt/d/AI_Seismic_Data/INSTANCE/INSTANCE_filtered.csv",
    "p_col": "trace_P_arrival_sample",
    "s_col": "trace_S_arrival_sample",
    "samples": 12000,
    "dtype": "float32",
    "layout": "CT",
    "channel_order": "ZNE",
}
COLUMNS = [
    "dataset",
    "trace_name",
    "path",
    "offset",
    "nbytes",
    "samples",
    "dtype",
    "layout",
    "channel_order",
    "p_sample",
    "s_sample",
    "p_win",
    "s_win",
    "start",
    "split",
    "source_id",
    "category",
]


def add_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the 60 s window start and labels in window coordinates."""
    frame = frame.copy()
    frame["p_sample"] = pd.to_numeric(frame["p_sample"], errors="coerce").fillna(-1).astype(np.int64)
    frame["s_sample"] = pd.to_numeric(frame["s_sample"], errors="coerce").fillna(-1).astype(np.int64)
    is_noise = frame["category"].astype(str).str.lower().str.startswith("noise")
    start = np.clip(frame["p_sample"] - 500, 0, frame["samples"] - 6000)
    start = start.where(~is_noise, 0)
    frame["start"] = start.astype(np.int64)
    frame["p_win"] = np.where(is_noise, -1, frame["p_sample"] - frame["start"])
    frame["s_win"] = np.where(is_noise, -1, frame["s_sample"] - frame["start"])
    return frame


def bucket_rows(spec: dict, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(spec["csv"], low_memory=False)
    frame = frame[frame["split"].isin(["train", "dev", "test"])].copy()
    frame["split"] = frame["split"].replace({"dev": "val"})
    frame["dataset"] = dataset
    frame["path"] = spec["path"]
    frame["samples"] = spec["samples"]
    frame["dtype"] = spec["dtype"]
    frame["layout"] = spec["layout"]
    frame["channel_order"] = spec["channel_order"]
    frame["p_sample"] = pd.to_numeric(frame[spec["p_col"]], errors="coerce")
    frame["s_sample"] = pd.to_numeric(frame[spec["s_col"]], errors="coerce")

    parts = frame["trace_name"].astype(str).str.split("$", n=1, expand=True)
    frame["_group"] = parts[0]
    frame["_index"] = parts[1].str.split(",", n=1).str[0].astype(int)

    offsets = {}
    with h5py.File(spec["path"], "r") as handle:
        data = handle["data"]
        for group in frame["_group"].unique():
            dataset_obj = data[group]
            item_bytes = int(np.prod(dataset_obj.shape[1:])) * dataset_obj.dtype.itemsize
            offsets[group] = (int(dataset_obj.id.get_offset()), item_bytes)
    frame["offset"] = [
        offsets[g][0] + int(i) * offsets[g][1]
        for g, i in zip(frame["_group"], frame["_index"])
    ]
    frame["nbytes"] = [offsets[g][1] for g in frame["_group"]]
    frame["source_id"] = frame.get("source_id", frame["trace_name"]).astype(str)
    frame["category"] = "event"
    return add_windows(frame)[COLUMNS]


def stead_rows(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["dataset"] = "stead"
    frame["path"] = STEAD["path"]
    frame["samples"] = STEAD["samples"]
    frame["nbytes"] = STEAD["nbytes"]
    frame["dtype"] = STEAD["dtype"]
    frame["layout"] = STEAD["layout"]
    frame["channel_order"] = STEAD["channel_order"]
    return add_windows(frame)[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stead", default="data/stead_manifest.csv")
    parser.add_argument("--diting", default="data/diting_index.csv")
    parser.add_argument("--out", default="data/multidomain_manifest.csv")
    args = parser.parse_args()

    frames = [stead_rows(args.stead), bucket_rows(TEXD, "txed"), bucket_rows(INSTANCE, "instance")]
    if Path(args.diting).exists():
        diting = pd.read_csv(args.diting, low_memory=False)
        diting["dataset"] = "diting"
        diting["layout"] = "TC"
        frames.append(add_windows(diting)[COLUMNS])
    else:
        print(f"warning: {args.diting} not found, building without DiTing")

    frame = pd.concat(frames, ignore_index=True)
    is_noise = frame["category"].astype(str).str.lower().str.startswith("noise")
    valid_labels = (
        (frame["p_win"] > 0)
        & (frame["s_win"] > frame["p_win"])
        & (frame["s_win"] < 6000)
    )
    frame = frame[
        (is_noise | valid_labels)
        & (frame["offset"] > 0)
        & (frame["nbytes"] > 0)
        & (frame["samples"] >= 6000)
    ].reset_index(drop=True)
    frame.to_csv(args.out, index=False)
    print(frame.groupby(["dataset", "split", "category"]).size())
    print(f"wrote {args.out} ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
