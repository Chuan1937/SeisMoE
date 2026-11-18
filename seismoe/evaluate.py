"""Evaluate SeisMoE and PhaseNet/EQTransformer baselines on an indexed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import SteadDataset
from .eval import evaluate_picks, summarise, top1_picks
from .models.seismoe import SeisMoE

THRESHOLDS = (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def load_model(args, device):
    if args.model == "seismoe":
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = ckpt["args"]
        channels = tuple(ckpt.get("channels", cfg.get("channels", (32, 64, 96, 128, 192, 256))))
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
        model.load_state_dict(ckpt["model"])
        return model.to(device).eval(), "seismoe"
    import seisbench.models as sbm

    if args.model == "phasenet":
        model = sbm.PhaseNet.from_pretrained(args.weights) if args.weights else sbm.PhaseNet(weights=None)
        return model.to(device).eval(), "phasenet"
    if args.model == "eqtransformer":
        model = (
            sbm.EQTransformer.from_pretrained(args.weights)
            if args.weights
            else sbm.EQTransformer(weights=None)
        )
        return model.to(device).eval(), "eqtransformer"
    raise ValueError(args.model)


def _reorder_components(x: torch.Tensor, model) -> torch.Tensor:
    """Reorder channels from STEAD's ENZ order to the model's expected order."""
    order = getattr(model, "component_order", None)
    if not order:
        return x
    order = "".join(order)
    if sorted(order) != sorted("ENZ"):
        return x
    index = [("ENZ").index(ch) for ch in order]
    if index == [0, 1, 2]:
        return x
    return x[:, index, :]


@torch.no_grad()
def predict(model, loader, device, kind, amp="bf16", tta_shifts=(0,)):
    if kind == "phasenet":
        labels = list(model.labels) if not isinstance(model.labels, str) else list(model.labels)
        p_idx, s_idx, n_idx = labels.index("P"), labels.index("S"), labels.index("N")
    elif kind == "eqtransformer":
        labels = list(model.labels)
        p_idx, s_idx, n_idx = labels.index("P"), labels.index("S"), labels.index("Detection")
    probs_p, probs_s, probs_n = [], [], []
    for x, _y in loader:
        x = _reorder_components(x.to(device, non_blocking=True), model)
        acc_p = acc_s = acc_n = None
        for shift in tta_shifts:
            xs = torch.roll(x, shifts=shift, dims=-1) if shift else x
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp == "bf16" and device.type == "cuda"):
                out = model(xs)
            if kind == "seismoe":
                prob = torch.sigmoid(out[0].float())
                pp, ss, nn = prob[:, 0], prob[:, 1], prob[:, 2]
            elif kind == "phasenet":
                prob = out.float()
                pp, ss, nn = prob[:, p_idx], prob[:, s_idx], prob[:, n_idx]
            else:  # eqtransformer returns a tuple in model.labels order
                pp, ss, nn = out[p_idx].float(), out[s_idx].float(), out[n_idx].float()
            if shift:
                pp = torch.roll(pp, shifts=-shift, dims=-1)
                ss = torch.roll(ss, shifts=-shift, dims=-1)
                nn = torch.roll(nn, shifts=-shift, dims=-1)
            acc_p = pp if acc_p is None else acc_p + pp
            acc_s = ss if acc_s is None else acc_s + ss
            acc_n = nn if acc_n is None else acc_n + nn
        k = len(tta_shifts)
        probs_p.append((acc_p / k).cpu().numpy())
        probs_s.append((acc_s / k).cpu().numpy())
        probs_n.append((acc_n / k).cpu().numpy())
    return (
        np.concatenate(probs_p),
        np.concatenate(probs_s),
        np.concatenate(probs_n),
    )


def sweep_thresholds(p, s, p_true, s_true, is_noise, tolerance=50, thresholds=THRESHOLDS):
    best = None
    for thr_p in thresholds:
        for thr_s in thresholds:
            result = evaluate_picks(p, s, p_true, s_true, is_noise, thr_p, thr_s)
            m = summarise(result, tolerance)
            score = 0.5 * (m["P_f1"] + m["S_f1"])
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "thr_p": thr_p,
                    "thr_s": thr_s,
                    **m,
                }
    return best


def make_picks(p, s, thr_p, thr_s):
    rows = []
    for i in range(len(p)):
        pi, si = top1_picks(p[i], s[i], thr_p, thr_s)
        rows.append(
            (
                pi,
                si,
                float(p[i][pi]) if pi is not None else np.nan,
                float(s[i][si]) if si is not None else np.nan,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["seismoe", "phasenet", "eqtransformer"], default="seismoe")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--weights", default="", help="seisbench pretrained weights, e.g. original")
    parser.add_argument("--data", default="data/stead_subset.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="results/eval")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-traces", type=int, default=0)
    parser.add_argument("--thr-p", type=float, default=0.3)
    parser.add_argument("--thr-s", type=float, default=0.3)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--tolerance", type=int, default=50)
    parser.add_argument("--noise-head", action="store_true")
    parser.add_argument("--noise-thr", type=float, default=0.8)
    parser.add_argument("--smooth-sigma", type=float, default=0.0)
    parser.add_argument("--norm", choices=["std", "max"], default="std")
    parser.add_argument("--tta", default="", help="comma-separated shifts, e.g. 0,-100,100")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.data)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.max_traces:
        df = df.sample(n=min(args.max_traces, len(df)), random_state=0).reset_index(drop=True)
    dataset = SteadDataset(df, train=False, norm=args.norm)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    model, kind = load_model(args, device)
    shifts = tuple(int(v) for v in args.tta.split(",")) if args.tta else (0,)
    p, s, n = predict(model, loader, device, kind, tta_shifts=shifts)
    if args.smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d

        p = gaussian_filter1d(p, args.smooth_sigma, axis=1)
        s = gaussian_filter1d(s, args.smooth_sigma, axis=1)
    p_true = dataset._p
    s_true = dataset._s
    is_noise = p_true <= 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "weights": args.weights,
        "ckpt": args.ckpt,
        "split": args.split,
        "n_traces": len(df),
        "n_events": int((~is_noise).sum()),
        "n_noise": int(is_noise.sum()),
    }

    if args.sweep:
        best = sweep_thresholds(p, s, p_true, s_true, is_noise, args.tolerance)
        print(
            f"best val: thr_p={best['thr_p']} thr_s={best['thr_s']} "
            f"P_f1={best['P_f1']:.4f} S_f1={best['S_f1']:.4f}"
        )
        payload["sweep"] = best
        thr_p, thr_s = best["thr_p"], best["thr_s"]
    else:
        thr_p, thr_s = args.thr_p, args.thr_s
    payload["thresholds"] = {"p": thr_p, "s": thr_s}

    result = evaluate_picks(
        p,
        s,
        p_true,
        s_true,
        is_noise,
        thr_p,
        thr_s,
        noise_head=n if args.noise_head else None,
        noise_thr=args.noise_thr,
    )
    for tol in (10, 20, 50):
        if f"tol_{tol}" in result:
            payload[f"tolerance_{tol}"] = summarise(result, tol)
            m = payload[f"tolerance_{tol}"]
            print(
                f"tol {tol/100:.2f}s  "
                f"P: F1={m['P_f1']:.4f} P={m['P_precision']:.4f} R={m['P_recall']:.4f} MAE={m['P_mae']:.4f}  "
                f"S: F1={m['S_f1']:.4f} P={m['S_precision']:.4f} R={m['S_recall']:.4f} MAE={m['S_mae']:.4f}"
            )

    picks = make_picks(p, s, thr_p, thr_s)
    picks_df = pd.DataFrame(
        picks, columns=["p_idx", "s_idx", "p_conf", "s_conf"]
    )
    picks_df.insert(0, "trace_name", df["trace_name"].values)
    picks_df["p_true"] = p_true
    picks_df["s_true"] = s_true
    picks_df["is_noise"] = is_noise
    picks_df.to_csv(out / f"picks_{args.split}.csv", index=False)
    with open(out / "metrics.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"saved {out/'metrics.json'}")


if __name__ == "__main__":
    main()
