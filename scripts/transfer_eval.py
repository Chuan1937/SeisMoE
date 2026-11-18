"""Evaluate SeisMoE / PhaseNet / EQTransformer on TEXD and INSTANCE.

The two transfer datasets store traces as ``(3, T)`` in **ZNE** order inside
``f['data'][bucket][index]``; :mod:`seismoe.transfer` converts them to the
canonical ENZ layout used during STEAD training.  Baseline models then only
need their own channel reorder and normalisation rule.

Examples
--------
>>> python scripts/transfer_eval.py --dataset txed --split dev \
...     --model phasenet --weights original --norm std --out results/txed_phasenet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from seismoe.eval import evaluate_picks, summarise
from seismoe.transfer import load_transfer, predict_transfer

DATASETS = {
    "txed": {
        "csv": "/mnt/d/AI_Seismic_Data/txed/Texd_filtered.csv",
        "h5": "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
        "p_col": "trace_p_arrival_sample",
        "s_col": "trace_s_arrival_sample",
        "category_col": "trace_category",
    },
    "instance": {
        "csv": "/mnt/d/AI_Seismic_Data/INSTANCE/INSTANCE_filtered.csv",
        "h5": "/mnt/d/AI_Seismic_Data/INSTANCE/INSTANCE.hdf5",
        "p_col": "trace_P_arrival_sample",
        "s_col": "trace_S_arrival_sample",
        "category_col": "trace_category",
    },
}

THRESHOLDS = (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def load_seismoe(ckpt: str, device):
    from seismoe.models.seismoe import SeisMoE

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


def load_baseline(name: str, weights: str, device):
    import seisbench.models as sbm

    if name == "phasenet":
        model = sbm.PhaseNet.from_pretrained(weights)
    else:
        model = sbm.EQTransformer.from_pretrained(weights)
    return model.to(device).eval()


def reorder_to_model(x: torch.Tensor, component_order: str) -> torch.Tensor:
    index = [("ENZ").index(c) for c in component_order]
    return x[:, index, :] if index != [0, 1, 2] else x


@torch.no_grad()
def predict_baseline(model, waves: np.ndarray, device, name: str, batch: int = 128):
    probs_p, probs_s = [], []
    labels = list(model.labels) if not isinstance(model.labels, str) else list(model.labels)
    if name == "phasenet":
        p_idx, s_idx = labels.index("P"), labels.index("S")
    else:
        p_idx, s_idx = labels.index("P"), labels.index("S")
    for start in range(0, len(waves), batch):
        x = torch.from_numpy(waves[start : start + batch]).to(device)
        x = reorder_to_model(x, "".join(model.component_order))
        out = model(x)
        if name == "phasenet":
            probs_p.append(out[:, p_idx].float().cpu().numpy())
            probs_s.append(out[:, s_idx].float().cpu().numpy())
        else:
            probs_p.append(out[p_idx].float().cpu().numpy())
            probs_s.append(out[s_idx].float().cpu().numpy())
    return np.concatenate(probs_p), np.concatenate(probs_s)


def sweep(p, s, p_true, s_true, is_noise, tolerance=50):
    best = None
    for thr_p in THRESHOLDS:
        for thr_s in THRESHOLDS:
            result = evaluate_picks(p, s, p_true, s_true, is_noise, thr_p, thr_s)
            m = summarise(result, tolerance)
            score = 0.5 * (m["P_f1"] + m["S_f1"])
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "thr_p": thr_p,
                    "thr_s": thr_s,
                    "tol_50": m,
                    "tol_20": summarise(result, 20),
                }
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="txed")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model", choices=["seismoe", "phasenet", "eqtransformer"], default="seismoe")
    parser.add_argument("--ckpt", default="runs/moe_650k_ft/best.pt")
    parser.add_argument("--weights", default="original")
    parser.add_argument("--channel-order", default="ZNE", help="channel order stored in the file")
    parser.add_argument("--norm", default="std", choices=["std", "max", "none"])
    parser.add_argument("--bandpass", default="", help="band-pass in Hz, e.g. 1,45")
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--thr-p", type=float, default=0.1)
    parser.add_argument("--thr-s", type=float, default=0.1)
    parser.add_argument("--out", default="results/transfer")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    band = None
    if args.bandpass:
        low, high = (float(v) for v in args.bandpass.split(","))
        band = (low, high)
    ds = load_transfer(
        cfg["csv"],
        cfg["h5"],
        split=args.split,
        limit=args.max_traces or None,
        channel_order=args.channel_order,
        bucket=True,
        norm=args.norm,
        bandpass_hz=band,
        p_col=cfg["p_col"],
        s_col=cfg["s_col"],
        category_col=cfg["category_col"],
    )
    print(
        f"{args.dataset}/{args.split}: {len(ds)} traces "
        f"({int((ds.p > 0).sum())} events, {int((ds.p <= 0).sum())} noise)"
    )

    if args.model == "seismoe":
        model = load_seismoe(args.ckpt, device)
        p, s = predict_transfer(model, ds, device, batch=args.batch, workers=args.workers)
    else:
        model = load_baseline(args.model, args.weights, device)
        p, s = predict_baseline(model, ds.waveforms, device, args.model, args.batch)

    is_noise = ds.p <= 0
    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "weights": args.weights if args.model != "seismoe" else "",
        "ckpt": args.ckpt if args.model == "seismoe" else "",
        "channel_order": args.channel_order,
        "norm": args.norm,
        "bandpass": args.bandpass,
        "n": len(ds),
        "n_events": int((~is_noise).sum()),
    }
    if args.sweep:
        best = sweep(p, s, ds.p, ds.s, is_noise)
        payload["sweep"] = best
        thr_p, thr_s = best["thr_p"], best["thr_s"]
        print(
            f"best: thr_p={thr_p} thr_s={thr_s} "
            f"P50={best['tol_50']['P_f1']:.4f} S50={best['tol_50']['S_f1']:.4f} "
            f"P20={best['tol_20']['P_f1']:.4f} S20={best['tol_20']['S_f1']:.4f}"
        )
    else:
        thr_p, thr_s = args.thr_p, args.thr_s
    payload["thresholds"] = {"p": thr_p, "s": thr_s}
    result = evaluate_picks(p, s, ds.p, ds.s, is_noise, thr_p, thr_s)
    for tol in (10, 20, 50):
        m = summarise(result, tol)
        payload[f"tol_{tol}"] = m
        print(
            f"tol {tol/100:.2f}s  "
            f"P F1={m['P_f1']:.4f} (P={m['P_precision']:.4f} R={m['P_recall']:.4f} "
            f"MAE={m['P_mae']:.4f})  "
            f"S F1={m['S_f1']:.4f} (P={m['S_precision']:.4f} R={m['S_recall']:.4f} "
            f"MAE={m['S_mae']:.4f})"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"saved {out/'metrics.json'}")


if __name__ == "__main__":
    main()
