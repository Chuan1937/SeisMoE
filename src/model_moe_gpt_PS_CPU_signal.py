'''
Author: chuanjun
Date: 2025-03-14 11:51:18
email: chuanjun1978@gmail.com
website-github: https://github.com/Chuan1937
LastEditTime: 2025-03-14 11:54:15
'''
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph
# from time_moe import TimeMoeConfig, TimeMoeAttention, TimeMoeSparseExpertsLayer, load_balancing_loss_func
from Auxiliary import (ResCNNBlock,
                        BiLSTMBlock, uniform,
                        LayerNormalization, FeedForward,
                        CustomLSTM, ActivationLSTMCell,
                        TimeMoeSparseExpertsLayer,
                        TimeMoeConfig,
                        load_balancing_loss_func,
                        TimeMoeAttention)


#============================================================
# 多尺度特征提取器（原 Encoder）
#============================================================
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, filters, kernel_sizes, in_samples):
        super().__init__()
        conv_layers = []
        pool_layers = []
        self.extra_pads = []
        current_samples = in_samples
        for in_c, out_c, k in zip([in_channels] + filters[:-1], filters, kernel_sizes):
            conv_layers.append(
                nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2)
            )
            pad_val = current_samples % 2
            self.extra_pads.append(pad_val)
            pool_layers.append(nn.MaxPool1d(2, padding=0))
            current_samples = (current_samples + pad_val) // 2

        self.convs = nn.ModuleList(conv_layers)
        self.pools = nn.ModuleList(pool_layers)

    def forward(self, x):
        x = x.to("cuda")
        for conv, pool, pad in zip(self.convs, self.pools, self.extra_pads):
            x = F.relu(conv(x))
            if pad:
                x = F.pad(x, (0, pad), mode="constant", value=-65504)
            x = pool(x)
        return x

#============================================================
# 多阶段重构器（原 Decoder）
#============================================================
class MultiStageReconstructor(nn.Module):
    def __init__(self, in_channels, filters, kernel_sizes, out_samples, original_compatible=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.orig_compat = original_compatible
        self.crop_indices = []
        cur_samples = out_samples
        for i, _ in enumerate(filters):
            pad = cur_samples % 2
            cur_samples = (cur_samples + pad) // 2
            if pad:
                self.crop_indices.append(len(filters) - 1 - i)
        conv_layers = []
        for in_c, out_c, k in zip([in_channels] + filters[:-1], filters, kernel_sizes):
            conv_layers.append(
                nn.Conv1d(in_c, out_c, kernel_size=k, padding=k//2)
            )
        self.convs = nn.ModuleList(conv_layers)

    def forward(self, x):
        x = x.to("cuda")
        for i, conv in enumerate(self.convs):
            x = self.up(x)
            if self.orig_compat:
                if i == 3:
                    x = x[:, :, 1:-1]
            else:
                if i in self.crop_indices:
                    x = x[:, :, :-1]
            x = F.relu(conv(x))
        return x

#============================================================
# CNN 残差块堆叠
#============================================================
class CNNResBlockStack(nn.Module):
    def __init__(self, kernel_list, out_channels, dropout_rate):
        super().__init__()
        layers = []
        for k in kernel_list:
            layers.append(ResCNNBlock(out_channels, k, dropout_rate))
        self.stack = nn.ModuleList(layers)

    def forward(self, x):
        x = x.to("cuda")
        for layer in self.stack:
            x = layer(x)
        return x

#============================================================
# 堆叠双向 LSTM
#============================================================
class StackedBiLSTM(nn.Module):
    def __init__(self, num_blocks, input_dim, dropout_rate, hidden_dim=16, original_compatible=False):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(
            BiLSTMBlock(input_dim, hidden_dim, dropout_rate, original_compatible=original_compatible)
        )
        for _ in range(num_blocks - 1):
            self.lstm_layers.append(
                BiLSTMBlock(hidden_dim, hidden_dim, dropout_rate, original_compatible=original_compatible)
            )

    def forward(self, x):
        x = x.to("cuda")
        for lstm in self.lstm_layers:
            x = lstm(x)
        return x

#============================================================
# MoE 全局上下文模块
#============================================================
class MoEGlobalContextModule(nn.Module):
    def __init__(self, input_dim, drop_rate, attention_width=None, eps=1e-5, moe_config=None):
        super().__init__()
        config = TimeMoeConfig(
            hidden_size=input_dim,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=6000,
            rope_theta=10000,
            attention_dropout=drop_rate,
            _attn_implementation="TimeMoeAttention"
        )
        self.flash_attn = TimeMoeAttention(config, layer_idx=0)
        self.norm_pre = LayerNormalization(input_dim)
        if moe_config is not None:
            self.moe_ffn = TimeMoeSparseExpertsLayer(moe_config)
        else:
            self.moe_ffn = FeedForward(input_dim, drop_rate)
        self.norm_post = LayerNormalization(input_dim)

    def forward(self, x, attention_mask=None, position_ids=None):
        x = x.to("cuda")
        batch, channels, seq_len = x.size()
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch, seq_len)
        x_t = x.permute(0, 2, 1).contiguous()
        attn_out, attn_weights, _ = self.flash_attn(
            hidden_states=x_t,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True,
            use_cache=False
        )
        y = x_t + attn_out
        y = self.norm_pre(y.permute(0, 2, 1).contiguous())
        y_ffn = y.permute(0, 2, 1).contiguous()
        if isinstance(self.moe_ffn, TimeMoeSparseExpertsLayer):
            ffn_out, router_logits = self.moe_ffn(y_ffn)
            ffn_out = ffn_out.permute(0, 2, 1).contiguous()
        else:
            ffn_out = self.moe_ffn(y_ffn)
            router_logits = None
            ffn_out = ffn_out.permute(0, 2, 1).contiguous()
        y2 = y + ffn_out
        y2 = self.norm_post(y2)
        return y2, attn_weights, router_logits

#============================================================
# 主模型 IMG1
#============================================================
class IMG1(nn.Module):
    def __init__(
        self,
        in_channels=3,
        in_samples=6000,
        classes=2,
        lstm_blocks=3,
        drop_rate=0.1,
        original_compatible=False,
        norm="std",
        use_moe=True,
        num_experts=32,
        num_experts_per_tok=4,
        intermediate_size=256,
        hidden_act="gelu",
        router_aux_loss_factor=0.01,
        **kwargs,
    ):
        super().__init__()
        self.in_samples = in_samples
        self.in_channels = in_channels
        self.classes = classes
        self.lstm_blocks = lstm_blocks
        self.drop_rate = drop_rate
        self.norm = norm
        self.filters = [8, 16, 16, 32, 32, 64, 64]
        self.kernel_sizes = [11, 9, 7, 7, 5, 5, 3]
        self.res_cnn_kernels = [3, 3, 3, 3, 2, 3, 2]
        self.use_moe = use_moe
        self.hidden_size = 16

        if intermediate_size is None:
            intermediate_size = 4 * self.hidden_size
        if use_moe:
            self.moe_config = TimeMoeConfig(
                hidden_size=self.hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                router_aux_loss_factor=router_aux_loss_factor
            )
        else:
            self.moe_config = None

        eps = 1e-7 if original_compatible else 1e-5
        self.original_compatible = original_compatible

        self.feature_extractor = MultiScaleFeatureExtractor(
            in_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples
        )

        self.context_dp0 = MoEGlobalContextModule(
            input_dim=64, drop_rate=self.drop_rate, eps=eps,
            moe_config=TimeMoeConfig(
                hidden_size=64,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                router_aux_loss_factor=router_aux_loss_factor
            )
        )
        self.context_dp1 = MoEGlobalContextModule(
            input_dim=64, drop_rate=self.drop_rate, eps=eps,
            moe_config=TimeMoeConfig(
                hidden_size=64,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                router_aux_loss_factor=router_aux_loss_factor
            )
        )

        self.cnn_res_stack = CNNResBlockStack(
            kernel_list=self.res_cnn_kernels,
            out_channels=self.filters[-1],
            dropout_rate=self.drop_rate
        )

        self.context_dp2 = MoEGlobalContextModule(
            input_dim=64, drop_rate=self.drop_rate, eps=eps,
            moe_config=TimeMoeConfig(
                hidden_size=64,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                router_aux_loss_factor=router_aux_loss_factor
            )
        )

        self.bi_lstm = StackedBiLSTM(
            num_blocks=self.lstm_blocks,
            input_dim=self.filters[-1],
            dropout_rate=self.drop_rate,
            original_compatible=original_compatible
        )

        self.context_d0 = MoEGlobalContextModule(
            input_dim=16, drop_rate=self.drop_rate, eps=eps,
            moe_config=TimeMoeConfig(
                hidden_size=16,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                router_aux_loss_factor=router_aux_loss_factor
            )
        )
        self.context_d = MoEGlobalContextModule(
            input_dim=16, drop_rate=self.drop_rate, eps=eps,
            moe_config=TimeMoeConfig(
                hidden_size=16,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                router_aux_loss_factor=router_aux_loss_factor
            )
        )

        pick_lstms, pick_attns, pick_lstm2s, pick_lstm3s, pick_recons, pick_convs = [], [], [], [], [], []
        self.dropout_layer = nn.Dropout(self.drop_rate)
        for _ in range(self.classes):
            lstm_branch = nn.LSTM(16, 16, bidirectional=False)
            pick_lstms.append(lstm_branch)
            pick_attns.append(
                MoEGlobalContextModule(
                    input_dim=16, drop_rate=self.drop_rate, eps=eps,
                    moe_config=TimeMoeConfig(
                        hidden_size=16,
                        num_experts=num_experts,
                        num_experts_per_tok=num_experts_per_tok,
                        intermediate_size=intermediate_size,
                        hidden_act=hidden_act,
                        router_aux_loss_factor=router_aux_loss_factor
                    )
                )
            )
            pick_lstm2s.append(
                StackedBiLSTM(
                    num_blocks=self.lstm_blocks,
                    input_dim=self.filters[1],
                    dropout_rate=self.drop_rate,
                    original_compatible=original_compatible
                )
            )
            pick_lstm3s.append(
                StackedBiLSTM(
                    num_blocks=self.lstm_blocks,
                    input_dim=self.filters[1],
                    dropout_rate=self.drop_rate,
                    original_compatible=original_compatible
                )
            )
            pick_recons.append(
                MultiStageReconstructor(
                    in_channels=16,
                    filters=self.filters[::-1],
                    kernel_sizes=self.kernel_sizes[::-1],
                    out_samples=in_samples,
                    original_compatible=original_compatible
                )
            )
            pick_convs.append(
                nn.Conv1d(
                    in_channels=self.filters[0],
                    out_channels=1,
                    kernel_size=11,
                    padding=5
                )
            )
        self.pick_lstms = nn.ModuleList(pick_lstms)
        self.pick_attns = nn.ModuleList(pick_attns)
        self.pick_lstm2s = nn.ModuleList(pick_lstm2s)
        self.pick_lstm3s = nn.ModuleList(pick_lstm3s)
        self.pick_recons = nn.ModuleList(pick_recons)
        self.pick_convs = nn.ModuleList(pick_convs)

    def forward(self, x, logits=False):
        x = x.to("cuda")
        x = x.permute(0, 2, 1).contiguous()
        assert x.ndim == 3 and x.shape[1:] == (self.in_channels, self.in_samples)

        # 手动调用 flatten_parameters 解决 LSTM 警告
        for lstm_branch in self.pick_lstms:
            lstm_branch.flatten_parameters()

        feat = self.feature_extractor(x)
        feat, _, router_dp0 = self.context_dp0(feat)
        feat, _, router_dp1 = self.context_dp1(feat)
        feat = self.cnn_res_stack(feat)
        feat, _, router_dp2 = self.context_dp2(feat)
        feat = self.bi_lstm(feat)
        feat, _, router_d0 = self.context_d0(feat)
        feat, _, router_d = self.context_d(feat)

        outputs = []
        for lstm_branch, attn_branch, lstm2_branch, lstm3_branch, recon_branch, conv_branch in zip(
            self.pick_lstms, self.pick_attns, self.pick_lstm2s, self.pick_lstm3s, self.pick_recons, self.pick_convs
        ):
            p = feat.permute(2, 0, 1).contiguous()
            p = lstm_branch(p)[0]
            p = self.dropout_layer(p)
            p = p.permute(1, 2, 0).contiguous()
            p, _, _ = attn_branch(p)
            p = lstm2_branch(p)
            p = self.dropout_layer(p)
            p = lstm3_branch(p)
            p = recon_branch(p)
            if logits:
                pred = conv_branch(p)
            else:
                pred = torch.sigmoid(conv_branch(p))
            outputs.append(pred.squeeze(1))

        # 收集 MoE 路由 logits 和对应的配置
        router_list = []
        moe_configs = [
            self.context_dp0.moe_ffn.config,
            self.context_dp1.moe_ffn.config,
            self.context_dp2.moe_ffn.config,
            self.context_d0.moe_ffn.config,
            self.context_d.moe_ffn.config
        ]
        if router_dp0 is not None:
            router_list.append((router_dp0, moe_configs[0]))
        if router_dp1 is not None:
            router_list.append((router_dp1, moe_configs[1]))
        if router_dp2 is not None:
            router_list.append((router_dp2, moe_configs[2]))
        if router_d0 is not None:
            router_list.append((router_d0, moe_configs[3]))
        if router_d is not None:
            router_list.append((router_d, moe_configs[4]))

        # 计算负载均衡损失
        balance_loss = torch.tensor(0.0, device=x.device)
        if router_list:
            balance_loss = sum(
                load_balancing_loss_func(
                    router_logits,
                    config.num_experts,
                    config.num_experts_per_tok,
                    attention_mask=None
                ) for router_logits, config in router_list
            ) / len(router_list)
            # 确保标量输出，处理 DataParallel 的 Gather
            balance_loss = balance_loss.mean() if balance_loss.ndim > 0 else balance_loss
            if torch.isnan(balance_loss).any() or torch.isinf(balance_loss).any():
                raise ValueError("NaN/Inf in balance_loss")

        return tuple(outputs), balance_loss