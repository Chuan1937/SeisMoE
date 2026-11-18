"""STEAD waveform dataset backed by the byte-offset index.

Each item is read with ``os.pread`` from ``/mnt/f/STEAD/STEAD.hdf5`` (a 72 kB
contiguous float32 block), normalised per channel and augmented.  No waveform
is written to local disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SAMPLES = 6000
CHANNELS = 3
WAVE_BYTES = SAMPLES * CHANNELS * 4


@dataclass
class AugmentConfig:
    shift_prob: float = 0.9
    max_shift: int = 300
    noise_prob: float = 0.5
    noise_alpha: tuple[float, float] = (0.02, 0.5)
    scale_prob: float = 0.5
    scale_range: tuple[float, float] = (0.5, 2.0)
    flip_prob: float = 0.5
    channel_drop_prob: float = 0.05
    gaussian_prob: float = 0.2
    gaussian_sigma: tuple[float, float] = (0.01, 0.2)


def gaussian_labels(p: int, s: int, sigma: float = 10.0) -> np.ndarray:
    """Build [P, S, noise] targets of shape (3, 6000)."""
    y = np.zeros((3, SAMPLES), dtype=np.float32)
    if p <= 0:
        y[2] = 1.0
        return y
    t = np.arange(SAMPLES, dtype=np.float32)
    inv = 1.0 / (2.0 * sigma * sigma)
    y[0] = np.exp(-((t - p) ** 2) * inv)
    y[1] = np.exp(-((t - s) ** 2) * inv)
    return y


def normalize(x: np.ndarray, mode: str = "std") -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    if mode == "max":
        scale = np.abs(x).max(axis=0, keepdims=True)
    else:
        scale = x.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return x / scale


class SteadDataset(Dataset):
    def __init__(
        self,
        manifest: str | pd.DataFrame,
        h5_path: str = "/mnt/f/STEAD/STEAD.hdf5",
        train: bool = False,
        augment: AugmentConfig | None = None,
        noise_bank: int = 512,
        seed: int = 0,
        limit: int | None = None,
        max_traces: int | None = None,
        label_sigma: float = 10.0,
        norm: str = "std",
    ):
        df = pd.read_csv(manifest) if isinstance(manifest, str) else manifest
        # noise traces are only useful for the auxiliary noise head / hard negatives
        if max_traces is not None and len(df) > max_traces:
            event = df[df["category"] == "event"]
            noise = df[df["category"] == "noise"]
            n_event = min(len(event), int(max_traces * 0.82))
            n_noise = min(len(noise), max_traces - n_event)
            df = pd.concat(
                [
                    event.sample(n=n_event, random_state=seed),
                    noise.sample(n=n_noise, random_state=seed),
                ]
            ).sort_index()
        if limit is not None:
            df = df.head(limit)

        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.train = train
        self.aug = augment or AugmentConfig()
        self.noise_bank_size = noise_bank
        self.seed = seed
        self.label_sigma = label_sigma
        self.norm = norm
        self._fd: int | None = None
        self._bank: np.ndarray | None = None
        self._rng: np.random.Generator | None = None
        self._offsets = self.df["offset"].to_numpy(dtype=np.int64)
        self._p = self.df["p_sample"].to_numpy(dtype=np.int32)
        self._s = self.df["s_sample"].to_numpy(dtype=np.int32)

    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------ I/O
    def _open(self) -> None:
        if self._fd is None:
            self._fd = os.open(self.h5_path, os.O_RDONLY)

    def _read(self, idx: int) -> np.ndarray:
        self._open()
        raw = os.pread(self._fd, WAVE_BYTES, int(self._offsets[idx]))
        if len(raw) != WAVE_BYTES:
            raise IOError(f"short waveform read at {self._offsets[idx]}")
        return np.frombuffer(raw, dtype="<f4").reshape(SAMPLES, CHANNELS).copy()

    def _load_noise_bank(self) -> None:
        if self._bank is not None or self.noise_bank_size <= 0:
            return
        noise_idx = np.flatnonzero(self._p <= 0)
        if len(noise_idx) == 0:
            self._bank = np.zeros((1, SAMPLES, CHANNELS), dtype=np.float32)
            return
        rng = np.random.default_rng(self.seed + 1)
        pick = rng.choice(noise_idx, size=min(self.noise_bank_size, len(noise_idx)), replace=False)
        bank = np.empty((len(pick), SAMPLES, CHANNELS), dtype=np.float32)
        for j, i in enumerate(pick):
            bank[j] = normalize(self._read(int(i)))
        self._bank = bank

    def set_worker_seed(self, worker_id: int) -> None:
        self._rng = np.random.default_rng(int(worker_id))
        self._open()
        if self.train:
            self._load_noise_bank()

    # ------------------------------------------------------------ transform
    def _augment(self, x: np.ndarray, p: int, s: int) -> tuple[np.ndarray, int, int]:
        cfg = self.aug
        rng = self._rng
        if self.train and rng.random() < cfg.shift_prob and p > 0:
            limit = min(cfg.max_shift, p - 1, SAMPLES - 1 - s)
            if limit > 0:
                shift = int(rng.integers(-limit, limit + 1))
                if shift:
                    x = np.roll(x, shift, axis=0)
                    p += shift
                    s += shift

        if self.train and self._bank is not None and rng.random() < cfg.noise_prob:
            noise = self._bank[int(rng.integers(len(self._bank)))]
            alpha = 10.0 ** (-rng.uniform(1.0, 2.5))  # ~10-25 dB SNR
            x = x + alpha * noise

        if self.train and rng.random() < cfg.scale_prob:
            x = x * rng.uniform(*cfg.scale_range)
        if self.train and rng.random() < cfg.flip_prob:
            x = -x
        if self.train and rng.random() < cfg.channel_drop_prob:
            x = x.copy()
            x[:, int(rng.integers(CHANNELS))] = 0.0
        if self.train and rng.random() < cfg.gaussian_prob:
            x = x + rng.normal(0.0, rng.uniform(*cfg.gaussian_sigma), size=x.shape).astype(np.float32)
        if self.train and p > 0 and rng.random() < 0.2:
            # random second noise burst in a quiet part of the trace (hard negative)
            pass
        return x, p, s

    def __getitem__(self, idx: int):
        x = normalize(self._read(idx), self.norm)
        p = int(self._p[idx])
        s = int(self._s[idx])
        if self.train:
            x, p, s = self._augment(x, p, s)
        y = gaussian_labels(p, s, sigma=self.label_sigma)
        return torch.from_numpy(x.T.copy()), torch.from_numpy(y)
