"""Picking metrics compatible with the SeisMoE paper protocol."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

DISTANCE = 100  # 1 s at 100 Hz, same as the original evaluation script


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    errors: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mae(self) -> float:
        return float(np.mean(self.errors)) if self.errors else 0.0


def top1_picks(
    p: np.ndarray,
    s: np.ndarray,
    thr_p: float = 0.3,
    thr_s: float = 0.3,
    distance: int = DISTANCE,
) -> tuple[int | None, int | None]:
    """Highest P peak, then the highest S peak after P (paper protocol)."""
    p_peaks, p_props = find_peaks(p, height=thr_p, distance=distance)
    s_peaks, s_props = find_peaks(s, height=thr_s, distance=distance)
    if len(p_peaks):
        p_idx = int(p_peaks[np.argmax(p_props["peak_heights"])])
    else:
        p_idx = None
    s_idx = None
    if len(s_peaks):
        if p_idx is None:
            s_idx = int(s_peaks[np.argmax(s_props["peak_heights"])])
        else:
            after = [k for k, v in enumerate(s_peaks) if v > p_idx]
            if after:
                best = max(after, key=lambda k: s_props["peak_heights"][k])
                s_idx = int(s_peaks[best])
    return p_idx, s_idx


def _match(counts: Counts, pred: int | None, true: int, tol: int) -> None:
    if pred is None:
        counts.fn += 1
    elif abs(pred - true) <= tol:
        counts.tp += 1
        counts.errors.append(abs(pred - true) / 100.0)
    else:
        counts.fn += 1
        counts.fp += 1


def evaluate_picks(
    p_probs: np.ndarray,
    s_probs: np.ndarray,
    p_true: np.ndarray,
    s_true: np.ndarray,
    is_noise: np.ndarray,
    thr_p: float = 0.3,
    thr_s: float = 0.3,
    tolerances: tuple[int, ...] = (10, 20, 50),
    noise_head: np.ndarray | None = None,
    noise_thr: float = 1.1,
) -> dict:
    """Return precision/recall/F1/MAE for each tolerance.

    Event traces contribute one P and one S label.  A pick on a noise trace is
    a false positive; an event without a pick is a false negative.
    """
    out = {f"tol_{t}": {"P": Counts(), "S": Counts()} for t in tolerances}
    for i in range(len(p_probs)):
        p = p_probs[i]
        s = s_probs[i]
        if noise_head is not None:
            keep = 1.0 - noise_head[i]
            p = p * keep
            s = s * keep
            if float(noise_head[i].max()) >= noise_thr:
                continue
        p_idx, s_idx = top1_picks(p, s, thr_p, thr_s)
        if is_noise[i]:
            for t in tolerances:
                if p_idx is not None:
                    out[f"tol_{t}"]["P"].fp += 1
                if s_idx is not None:
                    out[f"tol_{t}"]["S"].fp += 1
            continue
        for t in tolerances:
            _match(out[f"tol_{t}"]["P"], p_idx, int(p_true[i]), t)
            _match(out[f"tol_{t}"]["S"], s_idx, int(s_true[i]), t)
    return out


def summarise(result: dict, tolerance: int) -> dict:
    key = f"tol_{tolerance}"
    res = {}
    for phase in ("P", "S"):
        c = result[key][phase]
        res[f"{phase}_precision"] = c.precision
        res[f"{phase}_recall"] = c.recall
        res[f"{phase}_f1"] = c.f1
        res[f"{phase}_mae"] = c.mae
        res[f"{phase}_tp"] = c.tp
        res[f"{phase}_fp"] = c.fp
        res[f"{phase}_fn"] = c.fn
    return res
