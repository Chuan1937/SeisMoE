"""Multi-domain dataset backed by the unified byte-offset manifest.

One model, four datasets (STEAD / TEXD / INSTANCE / DiTing), all read with
``os.pread`` and preprocessed identically:

* canonical ENZ channel order,
* optional 1-45 Hz zero-phase band-pass (removes the large DiTing drift),
* 60 s window around P (5 s pre-P),
* per-channel standardisation,
* the same augmentation as STEAD training.

Training uses a virtual epoch plan that samples datasets with fixed weights,
so small target domains (TEXD, INSTANCE) are not drowned by STEAD/DiTing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset

from .dataset import AugmentConfig, gaussian_labels

SAMPLES = 6000
CHANNELS = 3
DTYPES = {"float32": "<f4", "float64": "<f8", "int32": "<i4"}

_BAND = butter(4, [1.0 / 50.0, 45.0 / 50.0], btype="band")


def _bandpass(x: np.ndarray) -> np.ndarray:
    return filtfilt(_BAND[0], _BAND[1], x, axis=0)


@dataclass
class DomainWeights:
    stead: float = 0.40
    diting: float = 0.50
    txed: float = 0.05
    instance: float = 0.05


class MultiDomainDataset(Dataset):
    def __init__(
        self,
        manifest: str | pd.DataFrame,
        split: str = "train",
        train: bool = False,
        epoch_size: int = 0,
        weights: DomainWeights | None = None,
        val_per_dataset: int = 1000,
        augment: AugmentConfig | None = None,
        seed: int = 0,
        label_sigma: float = 10.0,
        bandpass: bool = True,
        bandpass_domains: tuple[str, ...] | None = None,
        datasets: list[str] | None = None,
    ):
        frame = pd.read_csv(manifest, low_memory=False) if isinstance(manifest, str) else manifest.copy()
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if datasets:
            frame = frame[frame["dataset"].isin(datasets)].reset_index(drop=True)
        keep = [
            "dataset",
            "path",
            "offset",
            "nbytes",
            "samples",
            "dtype",
            "layout",
            "channel_order",
            "p_win",
            "s_win",
            "start",
        ]
        frame = frame[keep]
        if not train and val_per_dataset > 0:
            parts = []
            for _, part in frame.groupby("dataset"):
                n = min(val_per_dataset, len(part))
                parts.append(part.sample(n=n, random_state=seed))
            frame = pd.concat(parts, ignore_index=True) if parts else frame
        self.df = frame
        self.train = train
        self.aug = augment or AugmentConfig()
        self.label_sigma = label_sigma
        self.bandpass = bandpass
        self.bandpass_domains = set(bandpass_domains) if bandpass_domains else None
        self._rng = np.random.default_rng(seed)
        self._fds: dict[str, int] = {}

        self._domain_index = {
            name: part.index.to_numpy()
            for name, part in frame.groupby("dataset")
        }
        self.domains = sorted(self._domain_index)
        if train:
            weights = weights or DomainWeights()
            present = {name: getattr(weights, name, 0.0) for name in self.domains}
            total = sum(present.values())
            if total <= 0:
                present = {name: 1.0 for name in self.domains}
                total = len(present)
            probs = np.array([present[name] / total for name in self.domains])
            self.plan = []
            for _ in range(epoch_size):
                domain = self.domains[int(self._rng.choice(len(self.domains), p=probs))]
                rows = self._domain_index[domain]
                self.plan.append(int(rows[int(self._rng.integers(len(rows)))]))
            self._p = frame["p_win"].to_numpy(dtype=np.int32)
            self._s = frame["s_win"].to_numpy(dtype=np.int32)
        else:
            self.plan = list(frame.index)
            self._p = frame["p_win"].to_numpy(dtype=np.int32)
            self._s = frame["s_win"].to_numpy(dtype=np.int32)

    def __len__(self) -> int:
        return len(self.plan)

    def set_worker_seed(self, worker_id: int) -> None:
        self._rng = np.random.default_rng(int(worker_id))

    def _open(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def _read(self, index: int) -> tuple[np.ndarray, int, int]:
        row = self.df.loc[index]
        fd = self._open(str(row["path"]))
        nbytes = int(row["nbytes"])
        raw = os.pread(fd, nbytes, int(row["offset"]))
        if len(raw) != nbytes:
            raise IOError(f"short read at {row['offset']} ({len(raw)}/{nbytes})")
        x = np.frombuffer(raw, dtype=DTYPES[str(row["dtype"])])
        if str(row["layout"]) == "CT":
            x = x.reshape(CHANNELS, int(row["samples"])).T
        else:
            x = x.reshape(int(row["samples"]), CHANNELS)
        x = x.astype(np.float32)
        if str(row["channel_order"]) == "ZNE":
            x = x[:, [2, 1, 0]]
        apply_band = self.bandpass and (
            self.bandpass_domains is None or str(row["dataset"]) in self.bandpass_domains
        )
        if apply_band:
            x = _bandpass(x).astype(np.float32)
        p = int(row["p_win"])
        s = int(row["s_win"])
        start = int(row["start"])
        x = x[start : start + SAMPLES]
        if x.shape[0] < SAMPLES:
            x = np.pad(x, ((0, SAMPLES - x.shape[0]), (0, 0)))
        return x, p, s

    def _augment(self, x: np.ndarray, p: int, s: int) -> tuple[np.ndarray, int, int]:
        cfg = self.aug
        rng = self._rng
        if p > 0 and rng.random() < cfg.shift_prob:
            limit = min(cfg.max_shift, p - 1, SAMPLES - 1 - s)
            if limit > 0:
                shift = int(rng.integers(-limit, limit + 1))
                if shift:
                    x = np.roll(x, shift, axis=0)
                    p += shift
                    s += shift
        if rng.random() < cfg.scale_prob:
            x = x * rng.uniform(*cfg.scale_range)
        if rng.random() < cfg.flip_prob:
            x = -x
        if rng.random() < cfg.channel_drop_prob:
            x[:, int(rng.integers(CHANNELS))] = 0.0
        if rng.random() < cfg.gaussian_prob:
            x = x + rng.normal(0.0, rng.uniform(*cfg.gaussian_sigma), size=x.shape).astype(np.float32)
        return x, p, s

    def __getitem__(self, plan_index: int):
        for _ in range(5):
            index = self.plan[plan_index] if self.train else self.plan[plan_index]
            x, p, s = self._read(index)
            if p > 0 and not (0 < p < SAMPLES - 1 and p < s < SAMPLES):
                if not self.train:
                    break
                plan_index = int(self._rng.integers(len(self.plan)))
                continue
            x = x - x.mean(axis=0, keepdims=True)
            scale = x.std(axis=0, keepdims=True)
            if not np.isfinite(scale).all() or (scale < 1e-8).any():
                if not self.train:
                    break
                plan_index = int(self._rng.integers(len(self.plan)))
                continue
            x = x / scale
            if self.train:
                x, p, s = self._augment(x, p, s)
            y = gaussian_labels(p, s, sigma=self.label_sigma)
            return torch.from_numpy(x.T.copy()), torch.from_numpy(y)
        x = np.zeros((CHANNELS, SAMPLES), dtype=np.float32)
        y = gaussian_labels(-1, -1, sigma=self.label_sigma)
        return torch.from_numpy(x), torch.from_numpy(y)
