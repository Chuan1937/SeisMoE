"""Baselines on the held-out DiTing 2.0 test split.

DiTing 2.0 is 100 Hz.  PhaseNet ``diting`` was trained on DiTing 1.0 at 50 Hz,
so its input is down-sampled 2:1 and its output is interpolated back to 100 Hz
for a common evaluation.  The other weights are evaluated at 100 Hz.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, resample_poly

from seismoe.eval import evaluate_picks, summarise

BAND = butter(4, [1 / 50, 45 / 50], btype="band")
DTYPES = {"float32": "<f4", "float64": "<f8", "int32": "<i4"}
CONFIGS = {
    "phasenet-diting": ("phasenet", "diting", "50", "ZNE", "std"),
    "phasenet-original": ("phasenet", "original", "100", "ENZ", "std"),
    "phasenet-stead": ("phasenet", "stead", "100", "ZNE", "max"),
    "eqt-original": ("eqtransformer", "original", "100", "ZNE", "std"),
    "eqt-stead": ("eqtransformer", "stead", "100", "ZNE", "max"),
}
GRID = (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)


def read_rows(frame: pd.DataFrame, limit: int, seed: int) -> list[dict]:
    if limit and len(frame) > limit:
        frame = frame.sample(n=limit, random_state=seed)
    out = []
    fds: dict[str, int] = {}
    for row in frame.to_dict("records"):
        path = row["path"]
        fd = fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            fds[path] = fd
        raw = os.pread(fd, int(row["nbytes"]), int(row["offset"]))
        x = np.frombuffer(raw, dtype=DTYPES[row["dtype"]])
        if row["layout"] == "CT":
            x = x.reshape(3, int(row["samples"])).T
        else:
            x = x.reshape(int(row["samples"]), 3)
        x = x.astype(np.float32)
        if row["channel_order"] == "ZNE":
            x = x[:, [2, 1, 0]]  # canonical ENZ
        x = filtfilt(BAND[0], BAND[1], x, axis=0).astype(np.float32)
        start = int(row["start"])
        x = x[start : start + 6000]
        row = dict(row)
        row["waveform"] = x
        out.append(row)
    for fd in fds.values():
        os.close(fd)
    return out


def prepare(x: np.ndarray, order: str, norm: str) -> np.ndarray:
    if order == "ZNE":
        x = x[:, [2, 1, 0]]  # ENZ -> ZNE
    if norm == "max":
        scale = np.abs(x).max(axis=0, keepdims=True)
    else:
        scale = x.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (x / scale).T.astype(np.float32)


@torch.no_grad()
def predict(model, rows: list[dict], cfg, device, batch=128):
    name, weights, rate, order, norm = cfg
    probs_p, probs_s = [], []
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        waves = []
        for row in chunk:
            x = prepare(row["waveform"], order, norm)
            if rate == "50":
                x = resample_poly(x, 1, 2, axis=1)
            waves.append(x)
        x = torch.from_numpy(np.stack(waves)).to(device)
        out = model(x)
        if name == "phasenet":
            labels = list(model.labels)
            p_idx, s_idx = labels.index("P"), labels.index("S")
            p = out[:, p_idx].float().cpu().numpy()
            s = out[:, s_idx].float().cpu().numpy()
        else:
            labels = list(model.labels)
            p_idx, s_idx = labels.index("P"), labels.index("S")
            p = out[p_idx].float().cpu().numpy()
            s = out[s_idx].float().cpu().numpy()
        if rate == "50":
            grid = np.arange(6000) / 2.0
            src = np.arange(p.shape[1])
            p = np.stack([np.interp(grid, src, row) for row in p])
            s = np.stack([np.interp(grid, src, row) for row in s])
        probs_p.append(p)
        probs_s.append(s)
    return np.concatenate(probs_p), np.concatenate(probs_s)


def sweep(p, s, p_true, s_true, noise):
    best = None
    for tp in GRID:
        for ts in GRID:
            result = evaluate_picks(p, s, p_true, s_true, noise, tp, ts)
            m = summarise(result, 50)
            score = 0.5 * (m["P_f1"] + m["S_f1"])
            if best is None or score > best["score"]:
                best = {"score": score, "thr_p": tp, "thr_s": ts}
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/multidomain_manifest.csv")
    parser.add_argument("--max-val", type=int, default=1000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--models", default="phasenet-diting,phasenet-original,eqt-original")
    parser.add_argument("--out", default="results/diting_baselines")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.manifest, low_memory=False)
    frame = frame[frame["dataset"] == "diting"]
    val = read_rows(frame[frame["split"] == "val"], args.max_val, 0)
    test = read_rows(frame[frame["split"] == "test"], args.max_test, 0)
    print(f"val={len(val)} test={len(test)}")
    p_true = np.array([int(r["p_win"]) for r in test])
    s_true = np.array([int(r["s_win"]) for r in test])
    vp = np.array([int(r["p_win"]) for r in val])
    vs = np.array([int(r["s_win"]) for r in val])

    import seisbench.models as sbm

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for display in args.models.split(","):
        name, weights, rate, order, norm = CONFIGS[display]
        model = (
            sbm.PhaseNet.from_pretrained(weights)
            if name == "phasenet"
            else sbm.EQTransformer.from_pretrained(weights)
        ).to(device).eval()
        print(f"\n=== {display} ({rate} Hz, {order}, {norm}) ===")
        p, s = predict(model, val, CONFIGS[display], device, args.batch)
        best = sweep(p, s, vp, vs, vp <= 0)
        print(f"val thr={best['thr_p']}/{best['thr_s']}")
        p, s = predict(model, test, CONFIGS[display], device, args.batch)
        result = evaluate_picks(p, s, p_true, s_true, p_true <= 0, best["thr_p"], best["thr_s"])
        row = {"model": display, "rate": rate, "thr_p": best["thr_p"], "thr_s": best["thr_s"]}
        for tol in (10, 20, 50):
            m = summarise(result, tol)
            row[f"P_f1@{tol}"] = m["P_f1"]
            row[f"S_f1@{tol}"] = m["S_f1"]
            print(
                f"tol={tol/100:.2f}s P F1={m['P_f1']:.4f} S F1={m['S_f1']:.4f} "
                f"P MAE={m['P_mae']:.4f} S MAE={m['S_mae']:.4f}"
            )
        rows.append(row)
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out / "diting_test.csv", index=False)
    print(f"saved {out/'diting_test.csv'}")


if __name__ == "__main__":
    main()
