"""Probability ensemble of STEAD checkpoints (specialist / dense / routed).

Thresholds are selected on the validation split and then frozen for test.
The default models are the MoE specialist and the dense ablation, both trained
on raw STEAD waveforms.

Example::

    python scripts/ensemble_stead.py \
        --ckpts runs/moe_650k_ft/final.pt,runs/dense_650k/best.pt \
        --weights 0.6,0.4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from seismoe.data.dataset import SteadDataset
from seismoe.eval import evaluate_picks, summarise
from seismoe.models.seismoe import SeisMoE

GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


def build_model(ckpt: str, device):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = state["args"]
    channels = tuple(state.get("channels", cfg["channels"]))
    model = SeisMoE(
        channels=channels,
        dropout=cfg["dropout"],
        dense_layers=cfg["dense_layers"],
        moe_layers=cfg["moe_layers"],
        num_experts=cfg["experts"],
        top_k=cfg["top_k"],
        expert_hidden=cfg["expert_hidden"],
        aux_weight=cfg["aux_weight"],
    )
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, loader, device):
    probs_p, probs_s = [], []
    for x, _y in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _aux = model(x)
        prob = torch.sigmoid(logits.float())
        probs_p.append(prob[:, 0].cpu().numpy())
        probs_s.append(prob[:, 1].cpu().numpy())
    return np.concatenate(probs_p), np.concatenate(probs_s)


def sweep(p, s, p_true, s_true, noise):
    best = None
    for tp in GRID:
        for ts in GRID:
            result = evaluate_picks(p, s, p_true, s_true, noise, tp, ts)
            m = summarise(result, 50)
            score = 0.5 * (m["P_f1"] + m["S_f1"])
            if best is None or score > best["score"]:
                best = {"score": score, "thr_p": tp, "thr_s": ts, "P": m}
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/stead_manifest.csv")
    parser.add_argument("--ckpts", default="runs/moe_650k_ft/final.pt,runs/dense_650k/best.pt")
    parser.add_argument("--weights", default="")
    parser.add_argument("--max-val", type=int, default=3000)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="results/ensemble_stead")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.data)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    if args.max_val:
        val = val.sample(n=min(args.max_val, len(val)), random_state=0).reset_index(drop=True)
    if args.max_test:
        test = test.sample(n=min(args.max_test, len(test)), random_state=0).reset_index(drop=True)
    ckpts = args.ckpts.split(",")
    weights = [float(v) for v in args.weights.split(",")] if args.weights else [1.0] * len(ckpts)
    if len(weights) != len(ckpts):
        raise SystemExit("--weights must match --ckpts")

    ds_val = SteadDataset(val, train=False)
    ds_test = SteadDataset(test, train=False)
    dl_val = DataLoader(ds_val, batch_size=args.batch, num_workers=args.workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch, num_workers=args.workers)

    val_probs, test_probs = [], []
    for ckpt in ckpts:
        model = build_model(ckpt, device)
        print(f"predicting {ckpt} ...")
        val_probs.append(predict(model, dl_val, device))
        test_probs.append(predict(model, dl_test, device))
        del model
        torch.cuda.empty_cache()

    def mix(probs):
        p = sum(w * item[0] for w, item in zip(weights, probs)) / sum(weights)
        s = sum(w * item[1] for w, item in zip(weights, probs)) / sum(weights)
        return p, s

    p_val, s_val = mix(val_probs)
    best = sweep(p_val, s_val, ds_val._p, ds_val._s, ds_val._p <= 0)
    print(
        f"val thr_p={best['thr_p']} thr_s={best['thr_s']} "
        f"P50={best['P']['P_f1']:.4f} S50={best['P']['S_f1']:.4f}"
    )
    p_test, s_test = mix(test_probs)
    result = evaluate_picks(
        p_test, s_test, ds_test._p, ds_test._s, ds_test._p <= 0, best["thr_p"], best["thr_s"]
    )
    out = {
        "models": ckpts,
        "weights": weights,
        "thr_p": best["thr_p"],
        "thr_s": best["thr_s"],
        "n_test": len(ds_test),
    }
    for tol in (10, 20, 50):
        m = summarise(result, tol)
        out[f"tol_{tol}"] = m
        print(
            f"tol={tol/100:.2f}s  P F1={m['P_f1']:.4f} S F1={m['S_f1']:.4f} "
            f"P MAE={m['P_mae']:.4f} S MAE={m['S_mae']:.4f}"
        )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, default=float)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
