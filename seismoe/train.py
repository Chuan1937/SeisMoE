"""Train SeisMoE on the indexed STEAD subset."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import SteadDataset
from .eval import evaluate_picks, summarise
from .models.seismoe import SeisMoE


LOG_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "lr",
    "time",
    "P_precision",
    "P_recall",
    "P_f1",
    "P_mae",
    "S_precision",
    "S_recall",
    "S_f1",
    "S_mae",
    "P_f1_20",
    "S_f1_20",
]

TRANSFER_SPECS = {
    "txed": {
        "csv": "/mnt/d/AI_Seismic_Data/txed/Texd_filtered.csv",
        "h5": "/mnt/d/AI_Seismic_Data/txed/Texd.hdf5",
        "p_col": "trace_p_arrival_sample",
        "s_col": "trace_s_arrival_sample",
        "category_col": "trace_category",
        "train_split": "train",
        "val_split": "dev",
    },
    "instance": {
        "csv": "/mnt/d/AI_Seismic_Data/INSTANCE/INSTANCE_filtered.csv",
        "h5": "/mnt/d/AI_Seismic_Data/INSTANCE/INSTANCE.hdf5",
        "p_col": "trace_P_arrival_sample",
        "s_col": "trace_S_arrival_sample",
        "category_col": "trace_category",
        "train_split": "train",
        "val_split": "dev",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/stead_manifest.csv")
    p.add_argument("--out", default="runs/seismoe")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ema", type=float, default=0.997)
    p.add_argument("--noise-loss", type=float, default=0.2)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--val-thr", type=float, default=0.1)
    p.add_argument("--label-sigma", type=float, default=10.0)
    p.add_argument("--init", default="", help="load model weights only (fine-tuning)")
    p.add_argument(
        "--transfer",
        choices=["", "txed", "instance"],
        default="",
        help="fine-tune on a transfer dataset instead of STEAD",
    )
    p.add_argument("--multi", action="store_true", help="train one model on all four domains")
    p.add_argument("--multi-manifest", default="data/multidomain_manifest.csv")
    p.add_argument("--epoch-size", type=int, default=600_000)
    p.add_argument("--val-per-dataset", type=int, default=1000)
    p.add_argument(
        "--domain-weights",
        default="stead=0.40,diting=0.50,txed=0.05,instance=0.05",
    )
    p.add_argument("--no-bandpass", action="store_true")
    p.add_argument("--bandpass-domains", default="", help="comma-separated domains to band-pass; empty = all")
    p.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    p.add_argument("--channels", default="32,64,96,128,192,256")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--dense-layers", type=int, default=1)
    p.add_argument("--moe-layers", type=int, default=2)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--expert-hidden", type=int, default=512)
    p.add_argument("--aux-weight", type=float, default=0.01)
    p.add_argument("--resume", default="")
    p.add_argument("--tag", default="")
    return p.parse_args()


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for ema_param, param in zip(self.module.parameters(), model.parameters()):
            ema_param.mul_(d).add_(param.detach(), alpha=1.0 - d)
        for ema_buf, buf in zip(self.module.buffers(), model.buffers()):
            if ema_buf.dtype.is_floating_point:
                ema_buf.mul_(d).add_(buf.detach(), alpha=1.0 - d)
            else:
                ema_buf.copy_(buf)


def make_loader(dataset, batch_size, shuffle, workers, seed):
    def init_fn(worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        info.dataset.set_worker_seed(torch.initial_seed() % (2**32))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        worker_init_fn=init_fn,
        generator=torch.Generator().manual_seed(seed),
    )


def autocast_ctx(amp: str):
    if amp == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if amp == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cuda", enabled=False)


def batch_loss(logits, target, noise_weight, pos_weight=1.0):
    weights = torch.tensor([pos_weight, pos_weight, 1.0], device=logits.device, dtype=logits.dtype)
    p = F.binary_cross_entropy_with_logits(logits[:, 0], target[:, 0], pos_weight=weights[0])
    s = F.binary_cross_entropy_with_logits(logits[:, 1], target[:, 1], pos_weight=weights[1])
    n = F.binary_cross_entropy_with_logits(logits[:, 2], target[:, 2], pos_weight=weights[2])
    return p + s + noise_weight * n, p, s, n


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    noise_weight,
    amp,
    ds,
    tolerance=50,
    pos_weight=1.0,
    thr=0.1,
):
    model.eval()
    total, count = 0.0, 0
    probs_p, probs_s, probs_n = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_ctx(amp):
            logits, _aux = model(x)
            loss, *_ = batch_loss(logits.float(), y, noise_weight, pos_weight)
        total += float(loss) * len(x)
        count += len(x)
        prob = torch.sigmoid(logits.float())
        probs_p.append(prob[:, 0].cpu().numpy())
        probs_s.append(prob[:, 1].cpu().numpy())
        probs_n.append(prob[:, 2].cpu().numpy())
    probs_p = np.concatenate(probs_p)
    probs_s = np.concatenate(probs_s)
    probs_n = np.concatenate(probs_n)
    p_true = ds._p
    s_true = ds._s
    is_noise = p_true <= 0
    result = evaluate_picks(probs_p, probs_s, p_true, s_true, is_noise, thr, thr)
    metrics = summarise(result, tolerance)
    if tolerance != 20:
        m20 = summarise(result, 20)
        metrics["P_f1_20"] = m20["P_f1"]
        metrics["S_f1_20"] = m20["S_f1"]
    return total / max(count, 1), metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.multi:
        from .data.multidomain import DomainWeights, MultiDomainDataset

        weights = DomainWeights()
        for item in args.domain_weights.split(","):
            name, value = item.split("=")
            setattr(weights, name.strip(), float(value))
        train_ds = MultiDomainDataset(
            args.multi_manifest,
            split="train",
            train=True,
            epoch_size=args.epoch_size,
            weights=weights,
            seed=args.seed,
            label_sigma=args.label_sigma,
            bandpass=not args.no_bandpass,
            bandpass_domains=tuple(args.bandpass_domains.split(",")) if args.bandpass_domains else None,
        )
        val_ds = MultiDomainDataset(
            args.multi_manifest,
            split="val",
            train=False,
            val_per_dataset=args.val_per_dataset,
            seed=args.seed,
            label_sigma=args.label_sigma,
            bandpass=not args.no_bandpass,
            bandpass_domains=tuple(args.bandpass_domains.split(",")) if args.bandpass_domains else None,
        )
    elif args.transfer:
        from .transfer import TransferDataset, load_transfer

        spec = TRANSFER_SPECS[args.transfer]

        def load_split(split: str, limit: int | None):
            return load_transfer(
                spec["csv"],
                spec["h5"],
                split=split,
                limit=limit,
                channel_order="ZNE",
                bucket=True,
                norm="std",
                p_col=spec["p_col"],
                s_col=spec["s_col"],
                category_col=spec["category_col"],
            )

        train_set = load_split(spec["train_split"], args.max_train or None)
        val_set = load_split(spec["val_split"], args.max_val or None)
        train_ds = TransferDataset(
            train_set, train=True, seed=args.seed, label_sigma=args.label_sigma
        )
        val_ds = TransferDataset(
            val_set, train=False, seed=args.seed, label_sigma=args.label_sigma
        )
    else:
        train_df = __import__("pandas").read_csv(args.data)
        train_df = train_df[train_df["split"] == "train"]
        val_df = __import__("pandas").read_csv(args.data)
        val_df = val_df[val_df["split"] == "val"]
        train_ds = SteadDataset(
            train_df,
            train=True,
            seed=args.seed,
            max_traces=args.max_train or None,
            label_sigma=args.label_sigma,
        )
        val_ds = SteadDataset(
            val_df,
            train=False,
            seed=args.seed,
            max_traces=args.max_val or None,
            label_sigma=args.label_sigma,
        )
    print(f"train traces: {len(train_ds):,}  val traces: {len(val_ds):,}")

    train_loader = make_loader(
        train_ds, args.batch_size, True, args.workers, args.seed
    )
    val_loader = make_loader(
        val_ds, args.eval_batch, False, max(2, args.workers // 3), args.seed
    )

    channels = tuple(int(v) for v in args.channels.split(","))
    model = SeisMoE(
        channels=channels,
        dropout=args.dropout,
        dense_layers=args.dense_layers,
        moe_layers=args.moe_layers,
        num_experts=args.experts,
        top_k=args.top_k,
        expert_hidden=args.expert_hidden,
        aux_weight=args.aux_weight,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.2f}M")
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        print(f"initialised weights from {args.init}")

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.ndim <= 1 or "norm" in name or "bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.98),
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(model, args.ema) if args.ema > 0 else None

    start_epoch = 0
    best_score = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", -1.0)
        if ema is not None and ckpt.get("ema") is not None:
            ema.module.load_state_dict(ckpt["ema"])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    log_path = out / "log.csv"
    if start_epoch == 0:
        with open(log_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(LOG_FIELDS)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", ncols=110)
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_ctx(args.amp):
                logits, aux = model(x)
                loss, lp, ls, ln = batch_loss(logits.float(), y, args.noise_loss, args.pos_weight)
                loss = loss + aux
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            running += float(loss.detach()) * len(x)
            seen += len(x)
            progress.set_postfix(
                loss=f"{running/seen:.4f}",
                p=f"{float(lp):.4f}",
                s=f"{float(ls):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.1e}",
            )

        train_loss = running / max(seen, 1)
        elapsed = time.time() - t0
        metrics = {}
        val_loss = float("nan")
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            eval_model = ema.module if ema is not None else model
            val_loss, metrics = validate(
                eval_model,
                val_loader,
                device,
                args.noise_loss,
                args.amp,
                val_ds,
                pos_weight=args.pos_weight,
                thr=args.val_thr,
            )

        score = 0.5 * (
            metrics.get("P_f1_20", metrics.get("P_f1", 0.0))
            + metrics.get("S_f1_20", metrics.get("S_f1", 0.0))
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "time": elapsed,
            **metrics,
        }
        with open(log_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, extrasaction="ignore")
            writer.writerow(row)
        print(
            f"epoch {epoch}: train={train_loss:.5f} val={val_loss:.5f} "
            f"P_f1={metrics.get('P_f1', float('nan')):.4f} "
            f"S_f1={metrics.get('S_f1', float('nan')):.4f} "
            f"P_f1@20={metrics.get('P_f1_20', float('nan')):.4f} "
            f"S_f1@20={metrics.get('S_f1_20', float('nan')):.4f} "
            f"({elapsed:.0f}s)"
        )

        state = {
            "model": model.state_dict(),
            "ema": ema.module.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "args": vars(args),
            "channels": channels,
        }
        torch.save(state, out / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": (ema.module if ema is not None else model).state_dict(),
                    "args": vars(args),
                    "channels": channels,
                    "epoch": epoch,
                    "metrics": metrics,
                    "n_params": n_params,
                },
                out / "best.pt",
            )
            print(f"  saved best.pt (score={score:.4f})")

    print(f"done. best score={best_score:.4f}")


if __name__ == "__main__":
    main()
