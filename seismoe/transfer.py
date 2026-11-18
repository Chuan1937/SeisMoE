"""Zero-shot transfer evaluation on other HDF5 picking datasets.

Two layouts are supported:

* ``per-trace`` groups (STEAD/DiTing): ``f['data'][trace_name]``
* ``bucket`` groups (TEXD/INSTANCE): ``f['data'][group][index]`` where
  ``trace_name = "group$index,..."`` and the dataset is ``(3, T)``.

Traces are normalised per channel exactly like STEAD training.  Longer traces
(INSTANCE is 120 s) are cropped to a 60 s window containing both P and S.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, resample_poly
from torch.utils.data import DataLoader, Dataset

from .eval import evaluate_picks, summarise
from .data.dataset import AugmentConfig, gaussian_labels

SAMPLES = 6000
CHANNELS = 3


def normalize(x: np.ndarray, mode: str = "std") -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    if mode == "none":
        return x
    if mode == "max":
        scale = np.abs(x).max(axis=1, keepdims=True)
    else:
        scale = x.std(axis=1, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return x / scale


def bandpass(x: np.ndarray, low: float, high: float, fs: float = 100.0) -> np.ndarray:
    nyquist = 0.5 * fs
    b, a = butter(4, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, x, axis=1)


@dataclass
class TransferSet:
    waveforms: np.ndarray  # (N, 3, 6000) float32
    p: np.ndarray
    s: np.ndarray
    names: list[str]

    def __len__(self) -> int:
        return len(self.waveforms)


class ArrayDataset(Dataset):
    def __init__(self, data: TransferSet):
        self.data = data

    def __len__(self) -> int:
        return len(self.data.waveforms)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.data.waveforms[i]), 0


def _read_bucket(handle: h5py.File, trace_name: str) -> np.ndarray:
    group_name, rest = trace_name.split("$", 1)
    index = int(rest.split(",")[0])
    return np.asarray(handle["data"][group_name][index], dtype=np.float32)


def _read_trace(handle: h5py.File, trace_name: str) -> np.ndarray:
    data = np.asarray(handle["data"][trace_name][:], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"unexpected shape {data.shape} for {trace_name}")
    if data.shape[0] == CHANNELS:
        return data
    return data.T


def load_transfer(
    csv_path: str,
    h5_path: str,
    split: str = "test",
    limit: int | None = None,
    sample_rate: float = 100.0,
    channel_order: str = "ENZ",
    bucket: bool = False,
    norm: str = "std",
    bandpass_hz: tuple[float, float] | None = None,
    p_col: str = "trace_p_arrival_sample",
    s_col: str = "trace_s_arrival_sample",
    name_col: str = "trace_name",
    category_col: str = "trace_category",
    split_col: str = "split",
) -> TransferSet:
    df = pd.read_csv(csv_path, low_memory=False)
    if split and split_col in df.columns:
        df = df[df[split_col].astype(str).str.lower() == split.lower()]
    if limit:
        df = df.head(limit)
    waves, ps, ss, names = [], [], [], []
    order = [("ENZ").index(c) for c in channel_order]
    with h5py.File(h5_path, "r") as handle:
        for _, row in df.iterrows():
            name = str(row[name_col])
            try:
                data = _read_bucket(handle, name) if bucket else _read_trace(handle, name)
            except Exception:
                continue
            if data.shape[0] != CHANNELS:
                continue
            if abs(sample_rate - 100.0) > 1e-6:
                data = resample_poly(data, 100, int(round(sample_rate)), axis=1)
            data = data[order]
            if bandpass_hz is not None:
                data = bandpass(data, bandpass_hz[0], bandpass_hz[1])
            p = row.get(p_col, np.nan)
            s = row.get(s_col, np.nan)
            category = str(row.get(category_col, "event"))
            if category.lower().startswith("noise"):
                # noise window: take the first 60 s (or a centred crop)
                start = 0 if data.shape[1] <= SAMPLES else (data.shape[1] - SAMPLES) // 2
                window = data[:, start : start + SAMPLES]
                if window.shape[1] != SAMPLES:
                    continue
                waves.append(normalize(window, norm))
                ps.append(-1)
                ss.append(-1)
                names.append(name)
                continue
            if pd.isna(p) or pd.isna(s):
                # incomplete labels (e.g. INSTANCE P-only traces) are not noise
                continue
            p_i, s_i = int(p), int(s)
            if data.shape[1] > SAMPLES:
                # 60 s window that keeps both P and S; 5 s pre-P context.
                start = int(np.clip(p_i - 500, 0, data.shape[1] - SAMPLES))
                p_i -= start
                s_i -= start
                data = data[:, start : start + SAMPLES]
            if data.shape[1] < SAMPLES:
                pad = SAMPLES - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad)))
            if data.shape[1] != SAMPLES or not (0 < p_i < SAMPLES and 0 < s_i < SAMPLES):
                continue
            waves.append(normalize(data, norm))
            ps.append(p_i)
            ss.append(s_i)
            names.append(name)
    return TransferSet(
        np.stack(waves).astype(np.float32),
        np.asarray(ps, dtype=np.int64),
        np.asarray(ss, dtype=np.int64),
        names,
    )


class TransferDataset(Dataset):
    """Augmented view of an in-memory :class:`TransferSet`.

    Waveforms are already per-channel standardised by :func:`load_transfer`.
    The same geometric/amplitude augmentations as STEAD training are applied,
    except real-noise mixing (transfer sets have little or no noise traces).
    """

    def __init__(
        self,
        data: TransferSet,
        train: bool = False,
        augment: AugmentConfig | None = None,
        seed: int = 0,
        label_sigma: float = 10.0,
    ):
        self.waveforms = data.waveforms
        self._p = data.p.astype(np.int32)
        self._s = data.s.astype(np.int32)
        self.train = train
        self.aug = augment or AugmentConfig()
        self.label_sigma = label_sigma
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.waveforms)

    def set_worker_seed(self, worker_id: int) -> None:
        self._rng = np.random.default_rng(int(worker_id))

    def __getitem__(self, index: int):
        x = self.waveforms[index].copy()
        p = int(self._p[index])
        s = int(self._s[index])
        if self.train:
            rng = self._rng
            cfg = self.aug
            if p > 0 and rng.random() < cfg.shift_prob:
                limit = min(cfg.max_shift, p - 1, SAMPLES - 1 - s)
                if limit > 0:
                    shift = int(rng.integers(-limit, limit + 1))
                    if shift:
                        x = np.roll(x, shift, axis=1)
                        p += shift
                        s += shift
            if rng.random() < cfg.scale_prob:
                x = x * rng.uniform(*cfg.scale_range)
            if rng.random() < cfg.flip_prob:
                x = -x
            if rng.random() < cfg.channel_drop_prob:
                x[int(rng.integers(CHANNELS))] = 0.0
            if rng.random() < cfg.gaussian_prob:
                x = x + rng.normal(
                    0.0, rng.uniform(*cfg.gaussian_sigma), size=x.shape
                ).astype(np.float32)
        y = gaussian_labels(p, s, sigma=self.label_sigma)
        return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def predict_transfer(
    model,
    dataset: TransferSet,
    device,
    batch: int = 128,
    workers: int = 6,
):
    loader = DataLoader(ArrayDataset(dataset), batch_size=batch, num_workers=workers)
    probs_p, probs_s = [], []
    model.eval()
    for x, _ in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _aux = model(x)
        prob = torch.sigmoid(logits.float())
        probs_p.append(prob[:, 0].cpu().numpy())
        probs_s.append(prob[:, 1].cpu().numpy())
    return np.concatenate(probs_p), np.concatenate(probs_s)


@torch.no_grad()
def evaluate_transfer(
    model,
    dataset: TransferSet,
    device,
    thr_p: float = 0.1,
    thr_s: float = 0.1,
    batch: int = 128,
    workers: int = 6,
) -> dict:
    p, s = predict_transfer(model, dataset, device, batch, workers)
    is_noise = dataset.p <= 0
    result = evaluate_picks(p, s, dataset.p, dataset.s, is_noise, thr_p, thr_s)
    out = {"n": len(dataset), "n_noise": int(is_noise.sum())}
    for tol in (10, 20, 50):
        out[f"tol_{tol}"] = summarise(result, tol)
    return out
