"""Per-domain evaluation of the multi-domain SeisMoE checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from seismoe.data.multidomain import MultiDomainDataset
from seismoe.eval import evaluate_picks, summarise
from seismoe.models.seismoe import SeisMoE

GRID = (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)


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
def predict(model, dataset, device, batch=128, workers=6):
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)
    probs_p, probs_s = [], []
    for x, _y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _aux = model(x.to(device))
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
    parser.add_argument("--ckpt", default="runs/multidomain/best.pt")
    parser.add_argument("--manifest", default="data/multidomain_manifest.csv")
    parser.add_argument("--datasets", default="stead,diting,txed,instance")
    parser.add_argument("--max-val", type=int, default=1000)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bandpass", action="store_true", default=True)
    parser.add_argument("--bandpass-domains", default="")
    parser.add_argument("--out", default="results/multidomain")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.ckpt, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for domain in args.datasets.split(","):
        val = MultiDomainDataset(
            args.manifest,
            split="val",
            train=False,
            val_per_dataset=args.max_val,
            seed=0,
            bandpass=args.bandpass,
            bandpass_domains=tuple(args.bandpass_domains.split(",")) if args.bandpass_domains else None,
            datasets=[domain],
        )
        test = MultiDomainDataset(
            args.manifest,
            split="test",
            train=False,
            val_per_dataset=args.max_test,
            seed=0,
            bandpass=args.bandpass,
            bandpass_domains=tuple(args.bandpass_domains.split(",")) if args.bandpass_domains else None,
            datasets=[domain],
        )
        print(f"\n=== {domain}: val={len(val)} test={len(test)} ===")
        p, s = predict(model, val, device, args.batch, args.workers)
        best = sweep(p, s, val._p, val._s, val._p <= 0)
        print(
            f"val thr_p={best['thr_p']} thr_s={best['thr_s']} "
            f"P50={best['P']['P_f1']:.4f} S50={best['P']['S_f1']:.4f}"
        )
        p, s = predict(model, test, device, args.batch, args.workers)
        result = evaluate_picks(
            p, s, test._p, test._s, test._p <= 0, best["thr_p"], best["thr_s"]
        )
        row = {
            "dataset": domain,
            "n_test": len(test),
            "thr_p": best["thr_p"],
            "thr_s": best["thr_s"],
        }
        for tol in (10, 20, 50):
            m = summarise(result, tol)
            for key, value in m.items():
                row[f"{key}@{tol}"] = value
            print(
                f"test tol={tol/100:.2f}s  P F1={m['P_f1']:.4f} "
                f"(P={m['P_precision']:.4f} R={m['P_recall']:.4f} MAE={m['P_mae']:.4f})  "
                f"S F1={m['S_f1']:.4f} (P={m['S_precision']:.4f} "
                f"R={m['S_recall']:.4f} MAE={m['S_mae']:.4f})"
            )
        table.append(row)
        with open(out / f"{domain}.json", "w") as handle:
            json.dump(row, handle, indent=2, default=float)

    frame = pd.DataFrame(table)
    frame.to_csv(out / "per_domain_test.csv", index=False)
    print("\n", frame[["dataset", "P_f1@20", "S_f1@20", "P_f1@50", "S_f1@50"]].to_string(index=False))
    print(f"saved {out/'per_domain_test.csv'}")


if __name__ == "__main__":
    main()
