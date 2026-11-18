"""Snap P picks to STEAD's 1-second label grid and report the effect.

STEAD's P arrival labels are quantised: 94.5% lie within 5 samples of a whole
second.  EQTransformer's P output is 97% on this grid, while SeisMoE's raw
output is 78%.  This script applies the same convention post hoc to a saved
picks CSV and reports P F1 at 0.1/0.2/0.5 s before and after snapping.

Usage::

    python scripts/snap_picks.py results/final/picks_SeisMoE.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def f1(pred: np.ndarray, true: np.ndarray, noise: np.ndarray, tol: int) -> float:
    tp = fp = fn = 0
    for i in range(len(pred)):
        if noise[i]:
            if pred[i] >= 0:
                fp += 1
            continue
        if pred[i] < 0:
            fn += 1
        elif abs(pred[i] - true[i]) <= tol:
            tp += 1
        else:
            fp += 1
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("picks", help="CSV from seismoe.evaluate or final_compare")
    parser.add_argument("--grid", type=int, default=100, help="label grid in samples")
    args = parser.parse_args()

    frame = pd.read_csv(args.picks)
    raw = frame["p_idx"].fillna(-1).astype(int).to_numpy()
    true = frame["p_true"].astype(int).to_numpy()
    noise = frame["is_noise"].to_numpy(dtype=bool)

    snapped = raw.copy()
    has = raw >= 0
    snapped[has] = np.round(raw[has] / args.grid).astype(int) * args.grid

    print(f"{args.picks}: {len(frame):,} traces, {int((~noise).sum()):,} events")
    print("        raw      snapped")
    for tol in (10, 20, 50):
        print(
            f"P@{tol/100:.1f}s  {f1(raw, true, noise, tol):.4f}   "
            f"{f1(snapped, true, noise, tol):.4f}"
        )


if __name__ == "__main__":
    main()
