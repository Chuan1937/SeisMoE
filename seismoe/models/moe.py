"""Sparse mixture-of-experts building blocks (PyTorch, single GPU friendly)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A small position-wise feed-forward expert."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class SparseMoE(nn.Module):
    """Token-choice top-k MoE with a shared expert and load balancing.

    Args:
        dim: token width.
        num_experts: number of routed experts.
        top_k: experts activated per token.
        hidden: hidden width of each expert.
        shared: add a shared (always active) expert.
        jitter: multiplicative-free Gaussian router noise during training.
        aux_weight: weight of the Switch load-balancing loss.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        hidden: int = 512,
        shared: bool = True,
        dropout: float = 0.1,
        jitter: float = 0.01,
        aux_weight: float = 0.01,
        z_weight: float = 1e-3,
    ):
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter = jitter
        self.aux_weight = aux_weight
        self.z_weight = z_weight

        self.gate = nn.Linear(dim, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)
        self.experts = nn.ModuleList(
            [Expert(dim, hidden, dropout) for _ in range(num_experts)]
        )
        if shared:
            self.shared_expert = Expert(dim, hidden, dropout)
            self.shared_gate = nn.Linear(dim, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_gate = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, dim = x.shape
        tokens = batch * steps
        logits = self.gate(x)
        if self.training and self.jitter > 0:
            logits = logits + torch.randn_like(logits) * self.jitter

        probs = logits.float().softmax(dim=-1)
        top_w, top_i = probs.topk(self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)

        flat = x.reshape(tokens, dim)
        out = torch.zeros_like(flat)
        flat_w = top_w.reshape(tokens, self.top_k)
        flat_i = top_i.reshape(tokens, self.top_k)

        for expert_id, expert in enumerate(self.experts):
            has = flat_i == expert_id
            token_idx, slot_idx = has.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            value = expert(flat[token_idx])
            weight = flat_w[token_idx, slot_idx]
            out.index_add_(0, token_idx, value * weight[:, None])
        out = out.reshape(batch, steps, dim)

        if self.shared_expert is not None:
            gate = torch.sigmoid(self.shared_gate(x))
            out = out + gate * self.shared_expert(x)

        # Switch-Transformer load balancing (differentiable through probs).
        importance = probs.mean(dim=(0, 1))
        one_hot = F.one_hot(top_i.reshape(-1), self.num_experts).float()
        load = one_hot.mean(dim=0)
        aux = self.num_experts * (importance * load).sum()
        z = logits.float().logsumexp(dim=-1).pow(2).mean()
        return out, self.aux_weight * aux + self.z_weight * z
