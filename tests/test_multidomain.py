import os
import tempfile

import numpy as np
import pandas as pd
import torch

from seismoe.data.multidomain import MultiDomainDataset


def _write_case(root: str, layout: str, order: str):
    samples = 6000
    if layout == "TC":
        x = np.zeros((samples, 3), dtype=np.float32)
    else:
        x = np.zeros((3, samples), dtype=np.float32)
    p, s = 1200, 2400
    if layout == "TC":
        x[p, 0] = 10.0
        x[s, 1] = 10.0
    else:
        x[0, p] = 10.0
        x[1, s] = 10.0
    x += np.random.default_rng(0).normal(0.0, 0.01, size=x.shape).astype(np.float32)
    path = os.path.join(root, f"{layout}.bin")
    x.tofile(path)
    return path, samples, p, s


def test_multidomain_reads_tc_and_ct():
    with tempfile.TemporaryDirectory() as root:
        rows = []
        for layout in ("TC", "CT"):
            path, samples, p, s = _write_case(root, layout, "ENZ")
            rows.append(
                dict(
                    dataset=layout.lower(),
                    path=path,
                    offset=0,
                    nbytes=samples * 3 * 4,
                    samples=samples,
                    dtype="float32",
                    layout=layout,
                    channel_order="ENZ",
                    p_win=p,
                    s_win=s,
                    start=0,
                    split="val",
                    category="event",
                )
            )
        manifest = os.path.join(root, "manifest.csv")
        pd.DataFrame(rows).to_csv(manifest, index=False)
        ds = MultiDomainDataset(manifest, split="val", train=False, val_per_dataset=0, bandpass=False)
        assert len(ds) == 2
        for i in range(2):
            x, y = ds[i]
            assert x.shape == (3, 6000)
            assert y.shape == (3, 6000)
            assert int(y[0].argmax()) == rows[i]["p_win"]
            assert int(y[1].argmax()) == rows[i]["s_win"]
            assert torch.isfinite(x).all()


def test_multidomain_train_plan():
    with tempfile.TemporaryDirectory() as root:
        path, samples, p, s = _write_case(root, "TC", "ENZ")
        manifest = os.path.join(root, "manifest.csv")
        pd.DataFrame(
            [
                dict(
                    dataset="stead",
                    path=path,
                    offset=0,
                    nbytes=samples * 3 * 4,
                    samples=samples,
                    dtype="float32",
                    layout="TC",
                    channel_order="ENZ",
                    p_win=p,
                    s_win=s,
                    start=0,
                    split="train",
                    category="event",
                )
            ]
        ).to_csv(manifest, index=False)
        ds = MultiDomainDataset(manifest, split="train", train=True, epoch_size=10, bandpass=False)
        assert len(ds) == 10
        for i in range(3):
            x, y = ds[i]
            assert x.shape == (3, 6000)
