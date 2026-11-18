import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
import torch.optim as optim
from torchviz import make_dot
from transformers.utils import logging, is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10
from transformers import PreTrainedModel, Cache, DynamicCache, StaticCache
from transformers import PretrainedConfig
from typing import Optional, Tuple, List, Union
import math
import warnings

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    pass


class TimeMoeConfig(PretrainedConfig):
    model_type = "time_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 4096,
            intermediate_size: int = 22016,
            horizon_lengths: List[int] = 1,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: int = None,
            hidden_act: str = "silu",
            num_experts_per_tok: int = 2,
            num_experts: int = 1,
            max_position_embeddings: int = 32768,
            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-6,
            use_cache: bool = True,
            use_dense: bool = False,
            rope_theta: int = 10000,
            attention_dropout: float = 0.0,
            apply_aux_loss: bool = True,
            router_aux_loss_factor: float = 0.02,
            tie_word_embeddings: bool = False,
            **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        if isinstance(horizon_lengths, int):
            horizon_lengths = [horizon_lengths]
        self.horizon_lengths = horizon_lengths  # Predict horizon length for each prediction.
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_dense = use_dense
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.apply_aux_loss = apply_aux_loss
        self.router_aux_loss_factor = router_aux_loss_factor

        assert self.use_dense ^ self.apply_aux_loss, 'Both use_dense and apply_aux_loss cannot be set to True or False at the same time.'

        kwargs.pop('tie_word_embeddings', None)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def f1_Score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + 1e-7)  # Avoid division by zero
        return recall

    def precision(y_true, y_pred):
        true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + 1e-7)  # Avoid division by zero
        return precision

    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    f1_score = 2 * ((precision_val * recall_val) / (precision_val + recall_val + 1e-7))
    return f1_score


# def load_balancing_loss_func(
#         gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
#         top_k: int,
#         num_experts: int = None,
#         attention_mask: Optional[torch.Tensor] = None
# ) -> torch.Tensor:
#     r"""
#     Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

#     See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
#     function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
#     experts is too unbalanced.

#     Args:
#         gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor], List[torch.Tensor]):
#             Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
#             shape [batch_size X sequence_length, num_experts].
#         top_k (`int`)
#             Selected Top k over the experts.
#         attention_mask (`torch.Tensor`, None):
#             The attention_mask used in forward function
#             shape [batch_size X sequence_length] if not None.
#         num_experts (`int`, *optional*):
#             Number of experts

#     Returns:
#         The auxiliary loss.
#     """
#     if gate_logits is None or not isinstance(gate_logits, (tuple, list)) or gate_logits[0] is None:
#         return 0.0

#     compute_device = gate_logits[0].device
#     concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

#     routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

#     _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

#     expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

#     if attention_mask is None:
#         # Compute the percentage of tokens routed to each expert
#         tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

#         # Compute the average probability of routing to these experts
#         router_prob_per_expert = torch.mean(routing_weights, dim=0)
#     else:
#         batch_size, sequence_length = attention_mask.shape
#         num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

#         # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
#         expert_attention_mask = (
#             attention_mask[None, :, :, None, None]
#             .expand((num_hidden_layers, batch_size, sequence_length, 2, num_experts))
#             .reshape(-1, 2, num_experts)
#             .to(compute_device)
#         )

#         # Compute the percentage of tokens routed to each experts
#         tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
#             expert_attention_mask, dim=0
#         )

#         # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
#         router_per_expert_attention_mask = (
#             attention_mask[None, :, :, None]
#             .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
#             .reshape(-1, num_experts)
#             .to(compute_device)
#         )

#         # Compute the average probability of routing to these experts
#         router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
#             router_per_expert_attention_mask, dim=0
#         )

#     overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))

#     return overall_loss * num_experts
import torch
from typing import Union, Tuple, List, Optional


def load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
        top_k: int,
        num_experts: int = None,
        attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if gate_logits is None or not isinstance(gate_logits, (tuple, list)) or gate_logits[0] is None:
        return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).to(compute_device)

    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        attention_mask = attention_mask.to(compute_device)
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / (
                torch.sum(expert_attention_mask, dim=0) + 1e-9
        )

        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / (
                torch.sum(router_per_expert_attention_mask, dim=0) + 1e-9
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0).to(compute_device))

    return overall_loss * num_experts


# 学习率调度器
def _lr_schedule(epoch, lr):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * np.exp(0.1 * (10 - epoch))


def normalize(data, mode='std'):
    # 减去均值
    data = data - torch.mean(data, dim=0, keepdim=True)

    if mode == 'max':
        # 按最大值归一化
        max_data = torch.max(data, dim=0, keepdim=True)[0]
        max_data[max_data == 0] = 1  # 防止除以零
        data = data / max_data
    elif mode == 'std':
        # 按标准差归一化
        std_data = torch.std(data, dim=0, keepdim=True)
        std_data[std_data == 0] = 1  # 防止除以零
        data = data / std_data
    return data


class ResCNNBlock(nn.Module):
    def __init__(self, filters, ker, drop_rate):
        super().__init__()

        self.manual_padding = False
        if ker == 3:
            padding = 1
        else:
            # ker == 2
            # Manual padding emulate the padding in tensorflow
            self.manual_padding = True
            padding = 0

        self.dropout = SpatialDropout1d(drop_rate)

        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters, ker, padding=padding)

        self.norm2 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv2 = nn.Conv1d(filters, filters, ker, padding=padding)

    def forward(self, x):
        y = self.norm1(x)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv1(y)

        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv2(y)

        return x + y


class SpatialDropout1d(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = x.unsqueeze(dim=-1)  # Add fake dimension
        x = self.dropout(x)
        x = x.squeeze(dim=-1)  # Remove fake dimension
        return x


class BiLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, drop_rate, original_compatible=False):
        super().__init__()

        if original_compatible == "conservative":
            # The non-conservative model uses a sigmoid activiation as handled by the base nn.LSTM
            self.lstm = CustomLSTM(ActivationLSTMCell, input_size, hidden_size)
        elif original_compatible == "non-conservative":
            self.lstm = CustomLSTM(
                ActivationLSTMCell,
                input_size,
                hidden_size,
                gate_activation=torch.sigmoid,
            )
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(drop_rate)
        self.conv = nn.Conv1d(2 * hidden_size, hidden_size, 1)
        self.norm = nn.BatchNorm1d(hidden_size, eps=1e-3)

    def forward(self, x):
        x = x.permute(
            2, 0, 1
        )  # From batch, channels, sequence to sequence, batch, channels
        x = self.lstm(x)[0]
        x = self.dropout(x)
        x = x.permute(
            1, 2, 0
        )  # From sequence, batch, channels to batch, channels, sequence
        x = self.conv(x)
        x = self.norm(x)
        return x


class CustomLSTM(nn.Module):
    """
    LSTM to be used with custom cells
    """

    def __init__(self, cell, *cell_args, bidirectional=True, **cell_kwargs):
        super(CustomLSTM, self).__init__()
        self.cell_f = cell(*cell_args, **cell_kwargs)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.cell_b = cell(*cell_args, **cell_kwargs)

    def forward(self, input, state=None):
        # Forward
        state_f = state
        outputs_f = []
        for i in range(len(input)):
            out, state_f = self.cell_f(input[i], state_f)
            outputs_f += [out]

        outputs_f = torch.stack(outputs_f)

        if not self.bidirectional:
            return outputs_f, None

        # Backward
        state_b = state
        outputs_b = []
        l = input.shape[0] - 1
        for i in range(len(input)):
            out, state_b = self.cell_b(input[l - i], state_b)
            outputs_b += [out]

        outputs_b = torch.flip(torch.stack(outputs_b), dims=[0])

        output = torch.cat([outputs_f, outputs_b], dim=-1)

        # Keep second argument for consistency with PyTorch LSTM
        return output, None


def hard_sigmoid(x):
    return torch.clip(0.2 * x + 0.5, 0, 1)


class ActivationLSTMCell(nn.Module):
    """
    LSTM Cell using variable gating activation, by default hard sigmoid

    If gate_activation=torch.sigmoid this is the standard LSTM cell

    Uses recurrent dropout strategy from https://arxiv.org/abs/1603.05118 to match Keras implementation.
    """

    def __init__(
            self, input_size, hidden_size, gate_activation=hard_sigmoid, recurrent_dropout=0
    ):
        super(ActivationLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_activation = gate_activation
        self.recurrent_dropout = recurrent_dropout

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for param in [self.weight_hh, self.weight_ih]:
                for idx in range(4):
                    mul = param.shape[0] // 4
                    torch.nn.init.xavier_uniform_(param[idx * mul: (idx + 1) * mul])

    def forward(self, input, state):
        if state is None:
            hx = torch.zeros(
                input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype
            )
            cx = torch.zeros(
                input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype
            )
        else:
            hx, cx = state
        gates = (
                torch.mm(input, self.weight_ih.t())
                + self.bias_ih
                + torch.mm(hx, self.weight_hh.t())
                + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = self.gate_activation(ingate)
        forgetgate = self.gate_activation(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = self.gate_activation(outgate)

        if self.recurrent_dropout > 0:
            cellgate = F.dropout(cellgate, p=self.recurrent_dropout)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LayerNormalization(nn.Module):
    def __init__(self, filters, eps=1e-14):
        super().__init__()

        gamma = torch.ones(filters, 1)
        self.gamma = nn.Parameter(gamma)
        beta = torch.zeros(filters, 1)
        self.beta = nn.Parameter(beta)
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        var = torch.mean((x - mean) ** 2, 1, keepdim=True) + self.eps
        std = torch.sqrt(var)
        outputs = (x - mean) / std

        outputs = outputs * self.gamma
        outputs = outputs + self.beta

        return outputs


class FeedForward(nn.Module):
    def __init__(self, io_size, drop_rate, hidden_size=128):
        super().__init__()

        self.lin1 = nn.Linear(io_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, io_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # To (batch, time, channel)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        x = x.permute(0, 2, 1)  # To (batch, channel, time)

        return x


def uniform(a, b, *args):
    return a + (b - a) * torch.rand(*args)


class TimeMoeTemporalBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# class TimeMoeSparseExpertsLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.top_k = config.num_experts_per_tok
#         self.hidden_size = config.hidden_size
#         self.num_experts = config.num_experts
#         self.norm_topk_prob = False

#         moe_intermediate_size = self.config.intermediate_size // self.top_k

#         # gating
#         self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
#         self.experts = nn.ModuleList(
#             [TimeMoeTemporalBlock(
#                 hidden_size=self.config.hidden_size,
#                 intermediate_size=moe_intermediate_size,
#                 hidden_act=self.config.hidden_act,
#             ) for _ in range(self.num_experts)]
#         )

#         self.shared_expert = TimeMoeTemporalBlock(
#             hidden_size=self.config.hidden_size,
#             intermediate_size=self.config.intermediate_size,
#             hidden_act=self.config.hidden_act,
#         )
#         self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

#     def forward(self, hidden_states: torch.Tensor):
#         """ """
#         batch_size, sequence_length, hidden_dim = hidden_states.shape
#         # hidden_states = hidden_states.view(-1, hidden_dim)
#         hidden_states = hidden_states.reshape(-1, hidden_dim)
#         # router_logits -> (batch * sequence_length, n_experts)
#         router_logits = self.gate(hidden_states)

#         routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
#         routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
#         if self.norm_topk_prob:
#             routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
#         # 我们转换回input  dtype
#         routing_weights = routing_weights.to(hidden_states.dtype)

#         final_hidden_states = torch.zeros(
#             (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
#         )

#         # One 热编码选定的专家以创建专家掩码
#         # 这将用于轻松索引哪个 专家将被 sollicated
#         expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

#         # 遍历模型中的所有可用 专家并对每个 专家执行计算
#         for expert_idx in range(self.num_experts):
#             expert_layer = self.experts[expert_idx]
#             idx, top_x = torch.where(expert_mask[expert_idx])

#             #为正确的隐藏状态编制索引，并计算当前 EA 的 EA 隐藏状态。
#             # 我们需要确保将输出隐藏状态乘以相应 Token（top-1 和 top-2）上的 'routing_weights'
#             current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
#             current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

#             # However `index_add_` only support torch tensors for indexing so we'll use
#             # the `top_x` tensor here.
#             final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

#         shared_expert_output = self.shared_expert(hidden_states)
#         shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

#         final_hidden_states = final_hidden_states + shared_expert_output

#         final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
#         return final_hidden_states, router_logits
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeMoeSparseExpertsLayer(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.norm_topk_prob = False

        self.device = device if device is not None else torch.device("cuda")

        moe_intermediate_size = self.config.intermediate_size // self.top_k

        # Gating network
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False).to(self.device)

        # Experts (ModuleList of TimeMoeTemporalBlock)
        self.experts = nn.ModuleList([
            TimeMoeTemporalBlock(
                hidden_size=self.config.hidden_size,
                intermediate_size=moe_intermediate_size,
                hidden_act=self.config.hidden_act,
            ).to(self.device)
            for _ in range(self.num_experts)
        ])

        # Shared expert
        self.shared_expert = TimeMoeTemporalBlock(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
        ).to(self.device)

        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False).to(self.device)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: (batch_size, sequence_length, hidden_size)
        Returns:
            final_hidden_states: (batch_size, sequence_length, hidden_size)
            router_logits: (batch_size * sequence_length, num_experts)
        """
        device = hidden_states.device
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)  # [B*T, H]

        router_logits = self.gate(hidden_states)  # [B*T, num_experts]
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # [B*T, top_k]
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=device
        )

        # One-hot encode the selected experts
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0).to(device)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() == 0:
                continue

            current_state = hidden_states[top_x].reshape(-1, hidden_dim)  # [selected, H]
            expert_out = expert_layer(current_state) * routing_weights[top_x, idx, None]  # Scale by prob

            final_hidden_states.index_add_(0, top_x, expert_out.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))  # [B*T, 1]
        shared_expert_output = shared_gate * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->TimeMOE
# class TimeMoeRotaryEmbedding(torch.nn.Module):
#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
#         super().__init__()

#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#         # Build here to make `torch.jit.trace` work.
#         self._set_cos_sin_cache(
#             seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
#         )

#     def _set_cos_sin_cache(self, seq_len, device, dtype):
#         self.max_seq_len_cached = seq_len
#         t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

#         freqs = torch.outer(t, self.inv_freq)
#         # Different from paper, but it uses a different permutation in order to obtain the same calculation
#         emb = torch.cat((freqs, freqs), dim=-1)
#         self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
#         self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

#     def forward(self, x, seq_len=None):
#         # x: [bs, num_attention_heads, seq_len, head_size]
#         if seq_len > self.max_seq_len_cached:
#             self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

#         return (
#             self.cos_cached[:seq_len].to(dtype=x.dtype),
#             self.sin_cached[:seq_len].to(dtype=x.dtype),
#         )

import torch
import torch.nn as nn


class TimeMoeRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 默认device
        self.device = device if device is not None else torch.device("cuda")

        inv_freq = 1.0 / (
                    self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预先缓存cos/sin
        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=self.device,
                                dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        t = t.to(self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)

        self.register_buffer("cos_cached", emb.cos().to(dtype).to(device), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype).to(device), persistent=False)

    def forward(self, x, seq_len=None):
        # x: (bs, num_heads, seq_len, head_dim)
        device = x.device
        dtype = x.dtype

        if seq_len is None:
            seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            # 如果需要的长度超过缓存，重新构建
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)

        return cos, sin


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2Attention with Qwen2->TimeMoe
# class TimeMoeAttention(nn.Module):
#     """
#     Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
#     and "Generating Long Sequences with Sparse Transformers".
#     """

#     def __init__(self, config: TimeMoeConfig, layer_idx: Optional[int] = None):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx

#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.rope_theta = config.rope_theta
#         self.is_causal = True
#         self.attention_dropout = config.attention_dropout

#         if (self.head_dim * self.num_heads) != self.hidden_size:
#             raise ValueError(
#                 f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
#                 f" and `num_heads`: {self.num_heads})."
#             )
#         self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
#         self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
#         self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
#         self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

#         self.rotary_emb = TimeMoeRotaryEmbedding(
#             self.head_dim,
#             max_position_embeddings=self.max_position_embeddings,
#             base=self.rope_theta,
#         )

#     def forward(
#             self,
#             hidden_states: torch.Tensor,
#             attention_mask: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_value: Optional[Cache] = None,
#             output_attentions: bool = False,
#             **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if "padding_mask" in kwargs:
#             warnings.warn(
#                 "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
#             )
#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         kv_seq_len = key_states.shape[-2]
#         if past_key_value is not None:
#             if self.layer_idx is None:
#                 raise ValueError(
#                     f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
#                     "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
#                     "with a layer index."
#                 )
#             kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
#         cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

#         if past_key_value is not None:
#             cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         # repeat k/v heads if n_kv_heads < n_heads
#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

#         if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
#                 f" {attn_weights.size()}"
#             )

#         if attention_mask is not None:
#             if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
#                 )

#             attn_weights = attn_weights + attention_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#         attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value
class TimeMoeAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: TimeMoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = TimeMoeRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        # ======== 强制将所有输入搬到GPU上 ========
        device = torch.device('cuda')
        hidden_states = hidden_states.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        # past_key_value 需要你确认它内部也在 GPU，我假设它处理时自己内部管理
        # ======================================

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimeMoeFlashAttention2(TimeMoeAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:

            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        origin_dtype = query_states.dtype
        if origin_dtype not in [torch.bfloat16, torch.float16]:
            query_states = query_states.to(dtype=torch.bfloat16)
            key_states = key_states.to(dtype=torch.bfloat16)
            value_states = value_states.to(dtype=torch.bfloat16)

        # without attention mask to faster speed
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=causal
        )

        if origin_dtype not in [torch.bfloat16, torch.float16]:
            return attn_output.to(origin_dtype)
        else:
            return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


TIME_MOE_ATTENTION_CLASSES = {
    "eager": TimeMoeAttention,
}

# --------------------------------------------------------
# Swin Transformer 1D V2
# Copyright (c) 2024 Yukara Ikemiya
# Licensed under The MIT License [see LICENSE for details]
#
# The original code is borrowed from the following repository
# and modified
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    # copied from timm/models/layers/drop.py
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    # copied from timm/models/layers/drop.py
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        L (int): Length of data

    Returns:
        x: (B, L, C)
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    x = x.contiguous().view(B, L, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): The length of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (int): The length of the window in pre-training.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias=True, attn_drop=0., proj_drop=0., pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(1, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = relative_coords_w  # 2*W-1
        if pretrained_window_size > 0:
            relative_coords_table[:] /= (pretrained_window_size - 1)
        else:
            relative_coords_table[:] /= (self.window_size - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table.unsqueeze(1))  # (2*W-1, 1)

        # get pair-wise relative position index for each token inside the window
        coords_w = torch.arange(self.window_size)
        relative_coords = coords_w[:, None] - coords_w[None, :]  # W, W
        relative_coords[:, :] += self.window_size - 1  # shift to start from 0
        # relative_position_index | example
        # [2, 1, 0]
        # [3, 2, 1]
        # [4, 3, 2]
        self.register_buffer("relative_position_index", relative_coords)  # (W, W): range of 0 -- 2*(W-1)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # device = next(self.parameters()).device
        params = list(self.parameters())
        device = params[0].device if len(params) > 0 else torch.device("cpu")

        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table)  # (2*W-1, nH)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # (W, W, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, W, W
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=pretrained_window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, L, C = x.shape
        assert L >= self.window_size, f'input length ({L}) must be >= window size ({self.window_size})'
        assert L % self.window_size == 0, f'input length ({L}) must be divisible by window size ({self.window_size})'

        shortcut = x

        # zero-padding shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            shifted_x[:, -self.shift_size:] = 0.
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, C

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, L)  # (B, L, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
            x[:, :self.shift_size] = 0.  # remove invalid embs
        else:
            x = shifted_x

        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, L: int):
        flops = 0
        # norm1
        flops += self.dim * L
        # W-MSA/SW-MSA
        nW = L / self.window_size
        flops += nW * self.attn.flops(self.window_size)
        # mlp
        flops += 2 * L * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * L
        return flops


class SwinTransformerV2Layer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            window_size: int,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
            drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
            pretrained_window_size=0
    ):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}, num_heads={self.num_heads}, window_size={self.window_size}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


