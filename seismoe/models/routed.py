"""STEAD-routed decoder on top of the multi-domain SeisMoE backbone.

The joint model shares one decoder across STEAD, TEXD, INSTANCE and DiTing.
This wrapper freezes the shared encoder/context and the joint decoder, and
adds a second decoder/head specialised for STEAD.  One checkpoint, two decoder
paths: ``domain=0`` selects the STEAD path, anything else the joint path.

The STEAD decoder is initialised as a copy of the joint decoder, so the model
starts at the joint model's STEAD performance and can then specialise without
touching the other domains.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .seismoe import SeisMoE


class SteadRoutedModel(nn.Module):
    def __init__(self, base: SeisMoE):
        super().__init__()
        self.base = base
        self.stead_ups = copy.deepcopy(base.ups)
        self.stead_decoders = copy.deepcopy(base.decoders)
        self.stead_head = copy.deepcopy(base.head)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen backbone in eval mode so its BatchNorm statistics are
        # not updated by STEAD-only fine-tuning.
        self.base.eval()
        return self

    def forward(self, x: torch.Tensor, domain: int = 0):
        h, skips, aux = self.base.encode(x)
        if domain == 0:
            logits = self.base.decode(
                h, skips, self.stead_ups, self.stead_decoders, self.stead_head
            )
        else:
            logits = self.base.decode(
                h, skips, self.base.ups, self.base.decoders, self.base.head
            )
        return logits, aux
