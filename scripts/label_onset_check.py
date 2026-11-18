"""Check whether P picking errors come from coarse STEAD labels.

77% of STEAD P labels are multiples of 100 samples (one second), so a model
that predicts the physical onset is penalised by a 0.1/0.2 s metric.  This
script compares each model's P pick with an independent AIC onset measured on
the waveform, for traces where the pick disagrees with the label by more than
one 0.2 s tolerance.

Usage::

    python scripts/label_onset_check.py --ckpt runs/moe_650k_ft/best.pt \
        --max-traces 2000 --out results/label_onset.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d
from torch.utils.data import DataLoader

from seismoe.data.dataset import SteadDataset
from seismoe.eval import top1_picks
from seismoe.evaluate import load_model, predict


def aic_onset(envelope: np.ndarray) -> int:
    """AIC picker on a 1-D energy envelope (returns a local index)."""
    n = len(envelope)
    if n < 8:
        return int(np.argmax(np.diff(envelope))) + 1
    k = np.arange(1, n - 1)
    var1 = np.array([envelope[:i].var() for i in k]) + 1e-12
    var2 = np.array([envelope[i:].var() for i in k]) + 1e-12
    aic = k * np.log(var1) + (n - k - 1) * np.log(var2)
    return int(k[np.argmin(aic)])


def onset_for(ds: SteadDataset, index: int, label: int, half: int = 120) -> int:
    x = ds._read(index)
    envelope = np.sqrt((x.astype(np.float64) ** 2).mean(axis=1))
    envelope = uniform_filter1d(envelope, 9)
    low = max(0, label - half)
    high = min(len(envelope), label + half)
    return low + aic_onset(envelope[low:high])


def picks_for(model, kind, df, device, batch, workers):
    ds = SteadDataset(df, train=False)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    p, s, _ = predict(model, loader, device, kind)
    picks = []
    for i in range(len(df)):
        pi, _si = top1_picks(p[i], s[i], 0.02, 0.02)
        picks.append(pi if pi is not None else -1)
    return np.asarray(picks), ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/stead_manifest.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--ckpt", default="runs/moe_650k_ft/best.pt")
    parser.add_argument("--max-traces", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="results/label_onset.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.data)
    df = frame[frame["split"] == args.split]
    df = df.sample(n=min(args.max_traces, len(df)), random_state=0).reset_index(drop=True)

    models = [
        ("SeisMoE", SimpleNamespace(model="seismoe", ckpt=args.ckpt, weights="")),
        (
            "EQTransformer-original",
            SimpleNamespace(model="eqtransformer", ckpt="", weights="original"),
        ),
    ]
    picks = {}
    ds = None
    for name, ns in models:
        model, kind = load_model(ns, device)
        picks[name], ds = picks_for(model, kind, df, device, args.batch, args.workers)
        del model
        torch.cuda.empty_cache()
    assert ds is not None

    labels = ds._p
    ref = "SeisMoE"
    hard = []
    for i in range(len(df)):
        if labels[i] <= 0 or picks[ref][i] < 0:
            continue
        err = abs(int(picks[ref][i]) - int(labels[i]))
        if 10 < err <= 50:
            hard.append(i)

    rows = []
    for i in hard:
        label = int(labels[i])
        onset = onset_for(ds, i, label)
        row = {
            "index": i,
            "label": label,
            "quantized": bool(label % 100 == 0),
            "onset": int(onset),
            "err_label_onset": int(abs(label - onset)),
        }
        for name in picks:
            pick = int(picks[name][i]) if picks[name][i] >= 0 else -1
            row[f"{name}_pick"] = pick
            row[f"{name}_err_label"] = int(abs(pick - label)) if pick >= 0 else -1
            row[f"{name}_err_onset"] = int(abs(pick - onset)) if pick >= 0 else -1
        rows.append(row)

    out = {
        "split": args.split,
        "n_traces": len(df),
        "n_hard": len(rows),
        "quantized_share": float(np.mean(labels[labels > 0] % 100 == 0)),
        "hard_quantized_share": float(np.mean([r["quantized"] for r in rows])) if rows else 0.0,
    }
    for name in picks:
        err_label = np.array([r[f"{name}_err_label"] for r in rows], dtype=float)
        err_onset = np.array([r[f"{name}_err_onset"] for r in rows], dtype=float)
        valid = (err_label >= 0) & (err_onset >= 0)
        out[name] = {
            "mean_abs_pick_label": float(err_label[valid].mean()),
            "mean_abs_pick_onset": float(err_onset[valid].mean()),
            "pick_closer_to_onset": float(np.mean(err_onset[valid] < err_label[valid])),
            "pick_closer_to_label": float(np.mean(err_label[valid] < err_onset[valid])),
        }
        print(
            f"{name:24s} |pick-label|={out[name]['mean_abs_pick_label']:.1f} samples  "
            f"|pick-onset|={out[name]['mean_abs_pick_onset']:.1f}  "
            f"closer to onset {out[name]['pick_closer_to_onset']:.2%}"
        )

    print(
        f"\n{len(rows)} hard cases ({out['hard_quantized_share']:.1%} quantized labels), "
        f"mean |label-onset| = "
        f"{np.mean([r['err_label_onset'] for r in rows]):.1f} samples"
        if rows
        else "\nno hard cases"
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"summary": out, "cases": rows}, handle, indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
