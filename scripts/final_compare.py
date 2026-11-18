"""Final STEAD comparison with per-model normalisation and thresholds.

Every model sees exactly the same validation/test traces.  Thresholds are
tuned on validation (0.5 s tolerance) and then frozen for test, so no test
information leaks into the operating point.

The baseline preprocessing rules were established on the validation split:

===============  ==========  ==============  ==========================
model            weights     normalisation   channel order
===============  ==========  ==============  ==========================
PhaseNet         original    per-trace std   ENZ (stored order)
PhaseNet         stead       per-trace max   ZNE (reordered)
EQTransformer    original    per-trace std   ZNE (reordered)
EQTransformer    stead       per-trace max   ZNE (reordered)
===============  ==========  ==============  ==========================
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from seismoe.data.dataset import SteadDataset
from seismoe.eval import evaluate_picks, summarise, top1_picks
from seismoe.evaluate import load_model, predict, sweep_thresholds

CONFIGS = [
    # name, model, weights, norm
    ("SeisMoE", "seismoe", "", "std"),
    ("PhaseNet-original", "phasenet", "original", "std"),
    ("PhaseNet-stead", "phasenet", "stead", "max"),
    ("EQTransformer-original", "eqtransformer", "original", "std"),
    ("EQTransformer-stead", "eqtransformer", "stead", "max"),
]

GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


def probabilities(model, kind, df, device, batch, workers, norm, bandpass=False):
    ds = SteadDataset(df, train=False, norm=norm, bandpass=bandpass)
    loader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    p, s, n = predict(model, loader, device, kind)
    return p, s, n, ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/stead_manifest.csv")
    parser.add_argument("--ckpt", default="runs/moe_650k_ft/best.pt")
    parser.add_argument("--max-val", type=int, default=5000)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="results/final")
    parser.add_argument("--bandpass", action="store_true")
    parser.add_argument(
        "--models",
        default="seismoe,phasenet-original,phasenet-stead,eqtransformer-original,eqtransformer-stead",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.data)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    if args.max_val:
        val = val.sample(n=min(args.max_val, len(val)), random_state=0).reset_index(drop=True)
    if args.max_test:
        test = test.sample(n=min(args.max_test, len(test)), random_state=0).reset_index(drop=True)

    wanted = {value.lower() for value in args.models.split(",")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    payload = {}
    for display, model_name, weights, norm in CONFIGS:
        if display.lower() not in wanted and model_name not in wanted:
            continue
        print(f"\n=== {display} ({norm}) ===")
        ns = SimpleNamespace(model=model_name, ckpt=args.ckpt, weights=weights)
        model, kind = load_model(ns, device)

        p, s, n, ds = probabilities(model, kind, val, device, args.batch, args.workers, norm, args.bandpass)
        best = sweep_thresholds(
            p, s, ds._p, ds._s, ds._p <= 0, tolerance=50, thresholds=GRID
        )
        print(
            f"val thr_p={best['thr_p']} thr_s={best['thr_s']} "
            f"P50={best['P_f1']:.4f} S50={best['S_f1']:.4f}"
        )

        p, s, n, ds = probabilities(model, kind, test, device, args.batch, args.workers, norm, args.bandpass)
        result = evaluate_picks(
            p, s, ds._p, ds._s, ds._p <= 0, best["thr_p"], best["thr_s"]
        )
        row = {
            "model": display,
            "weights": weights,
            "norm": norm,
            "thr_p": best["thr_p"],
            "thr_s": best["thr_s"],
            "n_test": len(ds),
            "n_test_events": int((ds._p > 0).sum()),
            "val_P_f1": best["P_f1"],
            "val_S_f1": best["S_f1"],
        }
        for tol in (10, 20, 50):
            m = summarise(result, tol)
            row[f"P_f1@{tol}"] = m["P_f1"]
            row[f"S_f1@{tol}"] = m["S_f1"]
            row[f"P_prec@{tol}"] = m["P_precision"]
            row[f"P_rec@{tol}"] = m["P_recall"]
            row[f"S_prec@{tol}"] = m["S_precision"]
            row[f"S_rec@{tol}"] = m["S_recall"]
            row[f"P_mae@{tol}"] = m["P_mae"]
            row[f"S_mae@{tol}"] = m["S_mae"]
            print(
                f"test tol={tol/100:.2f}s  "
                f"P F1={m['P_f1']:.4f} (P={m['P_precision']:.4f} R={m['P_recall']:.4f} "
                f"MAE={m['P_mae']:.4f})  "
                f"S F1={m['S_f1']:.4f} (P={m['S_precision']:.4f} R={m['S_recall']:.4f} "
                f"MAE={m['S_mae']:.4f})"
            )
        picks = [
            top1_picks(p[i], s[i], best["thr_p"], best["thr_s"])
            for i in range(len(p))
        ]
        pick_df = pd.DataFrame(
            [(pi if pi is not None else -1, si if si is not None else -1) for pi, si in picks],
            columns=["p_idx", "s_idx"],
        )
        pick_df.insert(0, "trace_name", ds.df["trace_name"].values)
        pick_df["p_true"] = ds._p
        pick_df["s_true"] = ds._s
        pick_df["is_noise"] = ds._p <= 0
        pick_df.to_csv(out / f"picks_{display}.csv", index=False)

        rows.append(row)
        payload[display] = row

        del model
        gc.collect()
        torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison_test.csv", index=False)
    with open(out / "comparison.json", "w") as handle:
        json.dump(payload, handle, indent=2, default=float)

    print("\n| model | P F1@0.5 | S F1@0.5 | P F1@0.2 | S F1@0.2 | P MAE@0.5 |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['P_f1@50']:.4f} | {row['S_f1@50']:.4f} | "
            f"{row['P_f1@20']:.4f} | {row['S_f1@20']:.4f} | {row['P_mae@50']:.4f} |"
        )
    print(f"\nsaved {out/'comparison_test.csv'}")


if __name__ == "__main__":
    main()
