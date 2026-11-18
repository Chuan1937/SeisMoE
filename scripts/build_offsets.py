"""Resolve HDF5 byte offsets for every STEAD trace, fast.

The default HDF5 metadata cache is 32 MB, but the 1.26M-link group index is
~37 MB, so every name lookup re-reads it (~120 ms).  Bumping the metadata
cache to 128 MB (the build's maximum) keeps the index resident and turns
lookups into ~0.3 ms operations.  Only offsets are stored; no waveform data
is copied.
"""

from __future__ import annotations

import os
import time

import h5py
import numpy as np

H5 = "/mnt/f/STEAD/STEAD.hdf5"
NAMES = "data/stead_names.npy"
OUT = "data/stead_offsets.npy"


def open_fast(path: str) -> h5py.File:
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    cfg = fapl.get_mdc_config()
    cfg.max_size = 128 * 2**20
    cfg.initial_size = 64 * 2**20
    cfg.min_size = 2**20
    fapl.set_mdc_config(cfg)
    fid = h5py.h5f.open(path.encode(), h5py.h5f.ACC_RDONLY, fapl=fapl)
    return h5py.File(fid)


def main() -> None:
    t0 = time.time()
    if os.path.exists(NAMES):
        names = np.load(NAMES, allow_pickle=True).tolist()
        print(f"loaded {len(names):,} names")
    else:
        with open_fast(H5) as f:
            names = list(f["data"].keys())
        np.save(NAMES, np.array(names, dtype=object), allow_pickle=True)
        print(f"read {len(names):,} names in {time.time()-t0:.1f}s")

    offsets = np.zeros(len(names), dtype=np.uint64)
    with open_fast(H5) as f:
        group = f["data"]
        t1 = time.time()
        for i, name in enumerate(names):
            offsets[i] = group[name].id.get_offset()
            if (i + 1) % 100_000 == 0:
                elapsed = time.time() - t1
                print(
                    f"  {i+1:,}/{len(names):,}  {elapsed:.0f}s  "
                    f"{(i+1)/elapsed:,.0f}/s",
                    flush=True,
                )
        # verify
        rng = np.random.default_rng(0)
        sample = rng.choice(len(names), size=100, replace=False)
        bad = 0
        for i in sample:
            if int(offsets[i]) != group[names[int(i)]].id.get_offset():
                bad += 1
        print(f"verified 100 offsets, mismatches={bad}")

    np.save(OUT, offsets)
    print(f"wrote {OUT} ({len(offsets):,} offsets) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
