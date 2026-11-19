"""SeisMoE: residual U-Net with global attention and sparse MoE context.

The architecture keeps the two ingredients that make PhaseNet/EQTransformer
strong (multi-scale encoder/decoder with skip connections and a P/S/noise
multi-task head) and adds a global-context block at the bottleneck:

    conv stem -> 6 residual stages (6000 -> 188 samples)
              -> 1 dense transformer layer
              -> 2 sparse MoE transformer layers
              -> residual decoder with skips -> 3-channel logits

Only the bottleneck sees MoE: at 188 tokens a 3-channel 60 s trace has been
compressed by 32x, so tokens represent ~0.32 s patches and routing is a
semantic decision (P coda, S coda, noise, ...) rather than a local one.  This
keeps the MoE cheap and stable on a single 8 GB GPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import SparseMoE


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, act: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, dropout: float = 0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        return F.gelu(self.proj(x) + h)


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class SinusoidalPositions(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(x.dtype)


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer, FFN optionally replaced by a sparse MoE."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        ff: int = 512,
        dropout: float = 0.1,
        num_experts: int = 0,
        top_k: int = 2,
        expert_hidden: int = 512,
        shared_expert: bool = True,
        aux_weight: float = 0.01,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.is_moe = num_experts > 0
        if self.is_moe:
            self.moe = SparseMoE(
                dim,
                num_experts=num_experts,
                top_k=top_k,
                hidden=expert_hidden,
                shared=shared_expert,
                dropout=dropout,
                aux_weight=aux_weight,
            )
        else:
            self.ff = nn.Sequential(
                nn.Linear(dim, ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff, dim),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        h = self.norm2(x)
        if self.is_moe:
            out, aux = self.moe(h)
            return x + out, aux
        return x + self.ff(h), x.new_zeros(())


class SeisMoE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        samples: int = 6000,
        channels: tuple[int, ...] = (32, 64, 96, 128, 192, 256),
        dropout: float = 0.1,
        dense_layers: int = 1,
        moe_layers: int = 2,
        heads: int = 8,
        ff: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden: int = 512,
        shared_expert: bool = True,
        aux_weight: float = 0.01,
        out_channels: int = 3,
    ):
        super().__init__()
        self.samples = samples
        self.channels = channels

        self.stem = ConvNormAct(in_channels, channels[0], kernel=7)
        self.encoders = nn.ModuleList(
            [ResBlock(channels[0], dropout=dropout)]
            + [ResBlock(c, dropout=dropout) for c in channels[1:]]
        )
        self.downs = nn.ModuleList(
            [Downsample(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        dim = channels[-1]
        self.pos = SinusoidalPositions(dim)
        layers: list[TransformerLayer] = []
        for i in range(dense_layers):
            layers.append(
                TransformerLayer(dim, heads, ff, dropout, num_experts=0)
            )
        for i in range(moe_layers):
            layers.append(
                TransformerLayer(
                    dim,
                    heads,
                    ff,
                    dropout,
                    num_experts=num_experts,
                    top_k=top_k,
                    expert_hidden=expert_hidden,
                    shared_expert=shared_expert,
                    aux_weight=aux_weight,
                )
            )
        self.context = nn.ModuleList(layers)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        reversed_channels = list(channels[::-1])  # 256,192,128,96,64,32
        for i in range(len(channels) - 1):
            in_ch = reversed_channels[i]
            skip_ch = reversed_channels[i + 1]
            self.ups.append(ConvNormAct(in_ch, skip_ch, kernel=3))
            self.decoders.append(ResBlock(skip_ch * 2, skip_ch, dropout))

        self.head = nn.Sequential(
            ConvNormAct(channels[0], channels[0], kernel=3),
            nn.Conv1d(channels[0], out_channels, kernel_size=1),
        )
        self._init_weights()
        # Start with low phase probabilities and a moderate noise probability.
        with torch.no_grad():
            self.head[-1].bias.zero_()
            self.head[-1].bias[:2] = -4.0
            self.head[-1].bias[2] = -2.0

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor):
        """Encode to bottleneck features, skip list and auxiliary MoE loss."""
        h = self.stem(x)
        skips = []
        for i, encoder in enumerate(self.encoders):
            if i > 0:
                h = self.downs[i - 1](h)
            h = encoder(h)
            skips.append(h)

        tokens = h.permute(0, 2, 1)
        tokens = self.pos(tokens)
        aux = x.new_zeros(())
        for layer in self.context:
            tokens, layer_aux = layer(tokens)
            aux = aux + layer_aux
        h = tokens.permute(0, 2, 1)
        return h, skips, aux

    def decode(self, h, skips, ups, decoders, head) -> torch.Tensor:
        """Decode bottleneck features with a given decoder/head set."""
        for i, (up, decoder) in enumerate(zip(ups, decoders)):
            skip = skips[-2 - i]
            h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = up(h)
            h = decoder(torch.cat([h, skip], dim=1))
        return head(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, 3, T] -> logits [B, 3, T], auxiliary MoE loss."""
        h, skips, aux = self.encode(x)
        return self.decode(h, skips, self.ups, self.decoders, self.head), aux
