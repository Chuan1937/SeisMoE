"""Build a byte-offset index for the DiTing 2.0 HDF5 without copying waveforms.

DiTing 2.0 stores one contiguous ``(T, 3)`` int32 dataset per trace; T varies
between roughly 8.8k and 16k samples (100 Hz).  h5py name lookups on the 9p
mount are slow (~50/s single process), so offsets are resolved once with a
process pool and stored as metadata.  Training then reads each trace with
``os.pread``.

The metadata JSON provides Pg/Pn and Sg/Sn picks.  We use Pg (or Pn) and Sg
(or Sn), keep 0 < P < S < T and S-P < 50 s, and split by event id (the part
before ``_``) so traces of the same earthquake never cross splits.

Example::

    python scripts/build_diting_index.py --train 200000 --val 5000 --test 15000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

H5 = "/mnt/d/AI_Seismic_Data/谛听2.0/CENC_DiTingv2_natural_earthquake.hdf5"
JSON = "/mnt/d/AI_Seismic_Data/谛听2.0/CENC_DiTingv2_natural_earthquake.json"
OUT = "data/diting_index.csv"


def open_fast(path: str) -> h5py.File:
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    cfg = fapl.get_mdc_config()
    cfg.max_size = 128 * 2**20
    cfg.initial_size = 64 * 2**20
    cfg.min_size = 2**20
    fapl.set_mdc_config(cfg)
    return h5py.File(h5py.h5f.open(path.encode(), h5py.h5f.ACC_RDONLY, fapl=fapl))


def parse_labels(meta: dict) -> tuple[float, float] | None:
    p = meta.get("Pg") or meta.get("Pn")
    s = meta.get("Sg") or meta.get("Sn")
    try:
        p = float(p)
        s = float(s)
    except (TypeError, ValueError):
        return None
    if 0 < p < 9990 and p < s < 10000 and s - p < 5000:
        return p, s
    return None


def split_of(event: str) -> str:
    value = int(hashlib.md5(("diting:" + event).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if value < 0.10 else ("val" if value < 0.20 else "train")


def _lookup(args):
    h5_path, names = args
    handle = open_fast(h5_path)
    out = []
    for name in names:
        try:
            dataset = handle[name]
            out.append((name, int(dataset.id.get_offset()), int(dataset.nbytes)))
        except Exception:
            out.append((name, -1, -1))
    handle.close()
    return out


def resolve(names: list[str], h5_path: str, workers: int) -> dict[str, tuple[int, int]]:
    chunks = [(h5_path, names[i : i + 500]) for i in range(0, len(names), 500)]
    found: dict[str, tuple[int, int]] = {}
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for done, result in enumerate(pool.imap_unordered(_lookup, chunks), 1):
            for name, offset, nbytes in result:
                found[name] = (offset, nbytes)
            if done % 10 == 0 or done == len(chunks):
                elapsed = time.time() - t0
                total = min(done * 500, len(names))
                print(
                    f"  offsets {total:,}/{len(names):,}  {elapsed:.0f}s  "
                    f"{total / max(elapsed, 1e-9):.0f}/s",
                    flush=True,
                )
    return found


def _nonconstant_chunk(chunk):
    out = []
    for path, offset, nbytes in chunk:
        try:
            fd = os.open(path, os.O_RDONLY)
            raw = os.pread(fd, min(4096, nbytes), int(offset))
            os.close(fd)
            values = np.frombuffer(raw, dtype="<i4")
            out.append(float(values.std()) > 0)
        except Exception:
            out.append(False)
    return out


def check_nonconstant(rows: pd.DataFrame, workers: int = 8) -> pd.Series:
    """Read the first 4 kB of each trace and reject constant blocks."""
    payload = list(zip(rows["path"], rows["offset"], rows["nbytes"]))
    chunks = [payload[i : i + 1000] for i in range(0, len(payload), 1000)]
    flags = []
    with mp.Pool(workers) as pool:
        for result in pool.imap(_nonconstant_chunk, chunks):
            flags.extend(result)
    return pd.Series(flags, index=rows.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=200_000)
    parser.add_argument("--val", type=int, default=5_000)
    parser.add_argument("--test", type=int, default=15_000)
    parser.add_argument("--cap-per-event", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-constant", action="store_true")
    args = parser.parse_args()

    meta = json.load(open(JSON))
    targets = {"train": args.train, "val": args.val, "test": args.test}
    pools: dict[str, list[tuple[str, float, float]]] = {"train": [], "val": [], "test": []}
    for name, value in meta.items():
        labels = parse_labels(value)
        if labels is None:
            continue
        event = name.split("_")[0]
        pools[split_of(event)].append((name, *labels))
    for split in pools:
        print(f"{split}: {len(pools[split]):,} labelled traces")

    rng = np.random.default_rng(args.seed)
    selected = []
    for split, target in targets.items():
        by_event: dict[str, list[tuple[str, float, float]]] = {}
        for name, p, s in pools[split]:
            by_event.setdefault(name.split("_")[0], []).append((name, p, s))
        events = list(by_event)
        rng.shuffle(events)
        picked = []
        for event in events:
            traces = by_event[event]
            take = min(len(traces), args.cap_per_event)
            for j in rng.choice(len(traces), size=take, replace=False):
                picked.append((*traces[int(j)], split))
                if len(picked) >= target:
                    break
            if len(picked) >= target:
                break
        selected.extend(picked)
        print(f"{split}: picked {len(picked):,} traces from {len(events):,} events")

    frame = pd.DataFrame(selected, columns=["trace_name", "p_sample", "s_sample", "split"])
    frame["dataset"] = "diting"
    frame["path"] = H5
    frame["source_id"] = frame["trace_name"].str.split("_").str[0]
    frame["category"] = "event"

    print("resolving offsets ...")
    offsets = resolve(frame["trace_name"].tolist(), H5, args.workers)
    frame["offset"] = frame["trace_name"].map(lambda n: offsets.get(n, (-1, -1))[0])
    frame["nbytes"] = frame["trace_name"].map(lambda n: offsets.get(n, (-1, -1))[1])
    frame = frame[(frame["offset"] > 0) & (frame["nbytes"] > 0)].reset_index(drop=True)
    frame["samples"] = (frame["nbytes"] // 12).astype(np.int64)
    frame["dtype"] = "int32"
    frame["channel_order"] = "ZNE"

    if args.check_constant:
        print("checking constant traces ...")
        frame["nonconstant"] = check_nonconstant(frame, args.workers)
        frame = frame[frame["nonconstant"]].drop(columns=["nonconstant"]).reset_index(drop=True)

    # Keep traces for which a 60 s window around P/S exists.
    frame = frame[
        (frame["samples"] >= 6000)
        & (frame["p_sample"] < frame["samples"])
        & (frame["s_sample"] < frame["samples"])
    ].reset_index(drop=True)

    cols = [
        "dataset",
        "trace_name",
        "path",
        "offset",
        "nbytes",
        "samples",
        "dtype",
        "channel_order",
        "p_sample",
        "s_sample",
        "split",
        "source_id",
        "category",
    ]
    frame = frame[cols].sort_values(["split", "trace_name"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.groupby("split").size())
    print(f"wrote {args.out} ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
