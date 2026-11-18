import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ---------------------------------------------------------------------
# 1) We copy over any needed sub-layers from your "Auxiliary" or from
#    model_PS1.py, model_PS2.py. E.g.:
# ---------------------------------------------------------------------
from Auxiliary import (
    ResCNNBlock, BiLSTMBlock, LayerNormalization, FeedForward,
    CustomLSTM, ActivationLSTMCell, TimeMoeSparseExpertsLayer,
    TimeMoeConfig, load_balancing_loss_func,TimeMoeAttention
)

# ===== The following modules appear in model_PS1 / model_PS2 =====

class MultiScaleFeatureExtractor(nn.Module):
    """
    Same as in model_PS1: multi-scale conv+pool, reducing time dimension.
    """
    def __init__(self, in_channels, filters, kernel_sizes, in_samples):
        super().__init__()
        conv_layers = []
        pool_layers = []
        self.extra_pads = []
        current_samples = in_samples
        for in_c, out_c, k in zip([in_channels] + filters[:-1], filters, kernel_sizes):
            conv_layers.append(nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2))
            pad_val = current_samples % 2  # If odd length, pad on the right
            self.extra_pads.append(pad_val)
            pool_layers.append(nn.MaxPool1d(2, padding=0))
            current_samples = (current_samples + pad_val) // 2

        self.convs = nn.ModuleList(conv_layers)
        self.pools = nn.ModuleList(pool_layers)

    def forward(self, x):
        """
        x: [batch, in_channels, in_samples]
        """
        for conv, pool, pad in zip(self.convs, self.pools, self.extra_pads):
            x = F.relu(conv(x))
            if pad:
                x = F.pad(x, (0, pad), mode="constant", value=-1e10)
            x = pool(x)
        return x

class MultiStageReconstructor(nn.Module):
    """
    Same as in model_PS1 / model_PS2: upsampling + conv to expand back to original length.
    """
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
            conv_layers.append(nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2))

        self.convs = nn.ModuleList(conv_layers)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = self.up(x)
            if self.orig_compat:
                # Crop logic if desired
                if i == 3:
                    x = x[:, :, 1:-1]
            else:
                if i in self.crop_indices:
                    x = x[:, :, :-1]
            x = F.relu(conv(x))
        return x

class CNNResBlockStack(nn.Module):
    """
    Same as in model_PS1 / model_PS2: stack of ResCNNBlock
    """
    def __init__(self, kernel_list, out_channels, dropout_rate):
        super().__init__()
        layers = []
        for k in kernel_list:
            layers.append(ResCNNBlock(out_channels, k, dropout_rate))
        self.stack = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x

class StackedBiLSTM(nn.Module):
    """
    Same as in model_PS1 / model_PS2.
    """
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
        for lstm in self.lstm_layers:
            x = lstm(x)
        return x

class MoEGlobalContextModule(nn.Module):
    """
    Same as in model_PS1 / model_PS2: uses TimeMoeFlashAttention2 + feed-forward (MoE).
    """
    def __init__(self, input_dim, drop_rate, eps=1e-5, moe_config=None):
        super().__init__()
        config = TimeMoeConfig(
            hidden_size=input_dim,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=6000,
            rope_theta=10000,
            attention_dropout=drop_rate,
            _attn_implementation="flash_attention_2"
        )
        # self.flash_attn = TimeMoeFlashAttention2(config, layer_idx=0)
        self.flash_attn = TimeMoeAttention(config, layer_idx=0)
        self.norm_pre = LayerNormalization(input_dim)
        if moe_config is not None:
            self.moe_ffn = TimeMoeSparseExpertsLayer(moe_config)
        else:
            self.moe_ffn = FeedForward(input_dim, drop_rate)
        self.norm_post = LayerNormalization(input_dim)

    def forward(self, x, attention_mask=None, position_ids=None):
        """
        x: [batch, channels, seq_len]
        """
        batch, channels, seq_len = x.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch, seq_len)

        # Switch to [batch, seq_len, channels]
        x_t = x.permute(0, 2, 1)
        attn_out, attn_weights, router_logits = self.flash_attn(
            hidden_states=x_t,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True,
            use_cache=False
        )
        # residual + norm
        y = x_t + attn_out
        y = self.norm_pre(y.permute(0, 2, 1))

        # feed-forward
        y_ffn = y.permute(0, 2, 1)
        if isinstance(self.moe_ffn, TimeMoeSparseExpertsLayer):
            ffn_out, router_logits2 = self.moe_ffn(y_ffn)
            if router_logits2 is not None:
                router_logits = router_logits2
            ffn_out = ffn_out.permute(0, 2, 1)
        else:
            ffn_out = self.moe_ffn(y_ffn)
            ffn_out = ffn_out.permute(0, 2, 1)

        y2 = y + ffn_out
        y2 = self.norm_post(y2)

        return y2, attn_weights, router_logits

# ---------------------------------------------------------------------
# 2) P-Branch from Model_PS1
#    This replicates exactly the "picking" sub-network logic in model_PS1
# ---------------------------------------------------------------------
class PS1PickingBranch(nn.Module):
    """
    Single picking branch from Model_PS1.

    Typical steps in model_PS1's picking branch:
      1) Possibly a (Custom) LSTM
      2) MoE attention
      3) 2 stacked BiLSTMs
      4) Up-sample recon + final conv
    We'll replicate just one branch for the 'P wave.'
    """
    def __init__(self, in_dim=16, out_samples=6000, drop_rate=0.1,
                 filters=None, kernel_sizes=None,
                 original_compatible=False,
                 use_custom_lstm=False,
                 moe_config=None):
        super().__init__()
        if filters is None:
            filters = [8, 16, 16, 32, 32, 64, 64]
        if kernel_sizes is None:
            kernel_sizes = [11, 9, 7, 7, 5, 5, 3]

        self.dropout = nn.Dropout(drop_rate)

        # 1) LSTM branch from Model_PS1
        if use_custom_lstm:
            # E.g. the "if original_compatible == 'conservative': ... " part
            self.lstm_branch = CustomLSTM(ActivationLSTMCell, in_dim, in_dim, bidirectional=False)
        else:
            self.lstm_branch = nn.LSTM(in_dim, in_dim, bidirectional=False)

        # 2) Another MoEGlobalContextModule
        self.attn_branch = MoEGlobalContextModule(
            input_dim=in_dim,
            drop_rate=drop_rate,
            eps=1e-5,
            moe_config=moe_config
        )

        # 3) Two stacked BiLSTMs
        #   In Model_PS1 code, you had pick_lstm2s with input_dim = filters[1]=16, so let's do the same
        self.lstm2_branch = StackedBiLSTM(
            num_blocks=3,  # e.g. self.lstm_blocks
            input_dim=16,
            dropout_rate=drop_rate,
            original_compatible=original_compatible
        )
        self.lstm3_branch = StackedBiLSTM(
            num_blocks=3,
            input_dim=16,
            dropout_rate=drop_rate,
            original_compatible=original_compatible
        )

        # 4) Reconstructor + final conv => single channel
        self.recon = MultiStageReconstructor(
            in_channels=in_dim,
            filters=filters[::-1],
            kernel_sizes=kernel_sizes[::-1],
            out_samples=out_samples,
            original_compatible=original_compatible
        )
        self.final_conv = nn.Conv1d(
            in_channels=filters[0],  # which is 8
            out_channels=1,
            kernel_size=11,
            padding=5
        )

    def forward(self, feat, logits=False):
        """
        feat: [batch, channels=16, seq_length=?]
        Returns a 1D probability map: [batch, out_samples]
        """
        # 1) LSTM
        # Model_PS1 typically does -> p = feat.permute(2,0,1), run LSTM => permute back
        p = feat.permute(2, 0, 1)  # => [seq, batch, in_dim=16]
        p, _ = self.lstm_branch(p)  # => [seq, batch, 16]
        p = self.dropout(p)
        p = p.permute(1, 2, 0)      # => [batch, 16, seq]

        # 2) MoE attention
        p, _, _ = self.attn_branch(p)  # => [batch, 16, seq]

        # 3) Two stacked BiLSTMs
        p = self.lstm2_branch(p)
        p = self.dropout(p)
        p = self.lstm3_branch(p)

        # 4) Reconstruct
        p = self.recon(p)
        if logits:
            pred = self.final_conv(p)
        else:
            pred = torch.sigmoid(self.final_conv(p))

        # shape [batch, 1, out_samples]
        return pred.squeeze(1)  # => [batch, out_samples]

# ---------------------------------------------------------------------
# 3) S-Branch from Model_PS2
#    We take exactly the logic from model_PS2's picking branch for S
# ---------------------------------------------------------------------
class CRNNModule(nn.Module):
    """
    Exactly as in model_PS2
    """
    def __init__(self, in_channels, hidden_dim, kernel_size=3, num_lstm_layers=1, drop_rate=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)
        self.bi_lstm = nn.LSTM(in_channels, hidden_dim, num_layers=num_lstm_layers,
                               bidirectional=True, batch_first=True)
        self.proj = nn.Linear(2*hidden_dim, in_channels)

    def forward(self, x):
        # x: [batch, channels, seq]
        y = self.relu(self.bn(self.conv(x)))
        y = self.dropout(y)
        y = y.permute(0, 2, 1)  # => [batch, seq, channels]
        y, _ = self.bi_lstm(y)  # => [batch, seq, 2*hidden_dim]
        y = self.proj(y)        # => [batch, seq, in_channels]
        y = y.permute(0, 2, 1)  # => [batch, channels, seq]
        return y

class TCNModule(nn.Module):
    """
    Exactly as in model_PS2
    """
    def __init__(self, in_channels, num_layers=2, kernel_size=3, drop_rate=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = ((kernel_size - 1) * dilation) // 2
            conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation)
            layers += [
                conv,
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Dropout(drop_rate)
            ]
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        return self.tcn(x)

class PS2PickingBranch(nn.Module):
    """
    Single picking branch from model_PS2 code, e.g. for 'S wave.'
    Steps:
      1) A bidirectional LSTM
      2) CRNNModule
      3) TCNModule
      4) MoE attention
      5) 2 stacked BiLSTMs
      6) Reconstructor
      7) Final 1D conv => 1 channel
    """
    def __init__(self, in_dim=16, out_samples=6000, drop_rate=0.1,
                 filters=None, kernel_sizes=None,
                 original_compatible=False,
                 moe_config=None):
        super().__init__()
        if filters is None:
            filters = [8, 16, 16, 32, 32, 64, 64]
        if kernel_sizes is None:
            kernel_sizes = [11, 9, 7, 7, 5, 5, 3]

        self.dropout = nn.Dropout(drop_rate)

        # 1) BiLSTM => output dimension 16
        #   model_PS2 used nn.LSTM(16, 8, bidirectional=True) => yields [batch, seq, 16]
        self.lstm_branch = nn.LSTM(in_dim, 8, bidirectional=True)

        # 2) CRNNModule
        self.crnn = CRNNModule(in_channels=in_dim, hidden_dim=16,
                               kernel_size=3, num_lstm_layers=1, drop_rate=drop_rate)

        # 3) TCNModule
        self.tcn = TCNModule(in_channels=in_dim, num_layers=2, kernel_size=3, drop_rate=drop_rate)

        # 4) MoE attention
        self.attn_branch = MoEGlobalContextModule(
            input_dim=in_dim,
            drop_rate=drop_rate,
            eps=1e-5,
            moe_config=moe_config
        )

        # 5) 2 stacked BiLSTMs
        self.lstm2_branch = StackedBiLSTM(
            num_blocks=3,
            input_dim=filters[1],  # 16
            dropout_rate=drop_rate,
            original_compatible=original_compatible
        )
        self.lstm3_branch = StackedBiLSTM(
            num_blocks=3,
            input_dim=filters[1],
            dropout_rate=drop_rate,
            original_compatible=original_compatible
        )

        # 6) Reconstructor
        self.recon = MultiStageReconstructor(
            in_channels=in_dim,
            filters=filters[::-1],
            kernel_sizes=kernel_sizes[::-1],
            out_samples=out_samples,
            original_compatible=original_compatible
        )
        # 7) Final conv
        self.final_conv = nn.Conv1d(
            in_channels=filters[0],
            out_channels=1,
            kernel_size=11,
            padding=5
        )

    def forward(self, feat, logits=False):
        """
        feat: [batch, channels=16, seq_length=?]
        """
        # 1) BiLSTM
        p = feat.permute(2, 0, 1)  # => [seq, batch, 16]
        p, _ = self.lstm_branch(p) # => [seq, batch, 16]
        p = self.dropout(p)
        p = p.permute(1, 2, 0)     # => [batch, 16, seq]

        # 2) CRNN
        p = self.crnn(p)

        # 3) TCN
        p = self.tcn(p)

        # 4) MoE attention
        p, _, _ = self.attn_branch(p)

        # 5) two stacked BiLSTMs
        p = self.lstm2_branch(p)
        p = self.dropout(p)
        p = self.lstm3_branch(p)

        # 6) reconstruct
        p = self.recon(p)

        # 7) final conv => either raw logits or sigmoid
        if logits:
            pred = self.final_conv(p)
        else:
            pred = torch.sigmoid(self.final_conv(p))

        return pred.squeeze(1)

# ---------------------------------------------------------------------
# 4) The "Hybrid" model:
#    - shared trunk from model_PS1
#    - P wave = PS1PickingBranch
#    - S wave = PS2PickingBranch
# ---------------------------------------------------------------------
class IMG1(nn.Module):
    """
    A single trunk (taken from model_PS1) producing [batch, 16, reduced_seq],
    and then two separate picking heads:
      - self.p_branch = PS1PickingBranch (for P wave)
      - self.s_branch = PS2PickingBranch (for S wave)
    Returns: (p_pred, s_pred), plus a balancing loss if using MoE
    """
    def __init__(self,
                 in_channels=3,
                 in_samples=6000,
                 drop_rate=0.1,
                 original_compatible=False,
                 use_moe=True,
                 num_experts=32,
                 num_experts_per_tok=4,
                 intermediate_size=256,
                 hidden_act="gelu",
                 router_aux_loss_factor=0.01,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = in_samples
        self.drop_rate = drop_rate
        self.original_compatible = original_compatible

        # Example trunk filters (like model_PS1)
        self.filters = [8, 16, 16, 32, 32, 64, 64]
        self.kernel_sizes = [11, 9, 7, 7, 5, 5, 3]
        self.res_cnn_kernels = [3, 3, 3, 3, 2, 3, 2]

        # Construct trunk
        self.feature_extractor = MultiScaleFeatureExtractor(
            in_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples
        )
        # Two MoE blocks at 64 channels
        moe_cfg_64 = TimeMoeConfig(
            hidden_size=64,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=4*64 if intermediate_size is None else intermediate_size,
            hidden_act=hidden_act,
            router_aux_loss_factor=router_aux_loss_factor
        )
        self.context_dp0 = MoEGlobalContextModule(64, drop_rate, eps=1e-5, moe_config=moe_cfg_64)
        self.context_dp1 = MoEGlobalContextModule(64, drop_rate, eps=1e-5, moe_config=moe_cfg_64)

        # Optional CNN res stack => 64 channels
        self.cnn_res_stack = CNNResBlockStack(
            kernel_list=self.res_cnn_kernels,
            out_channels=self.filters[-1],  # 64
            dropout_rate=self.drop_rate
        )
        # Then a BiLSTM => 16 channels
        self.bi_lstm = StackedBiLSTM(
            num_blocks=3,
            input_dim=64,
            dropout_rate=self.drop_rate,
            original_compatible=original_compatible
        )
        # Then two more MoE blocks at 16 channels
        moe_cfg_16 = TimeMoeConfig(
            hidden_size=16,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=4*16 if intermediate_size is None else intermediate_size,
            hidden_act=hidden_act,
            router_aux_loss_factor=router_aux_loss_factor
        )
        self.context_d0 = MoEGlobalContextModule(16, drop_rate, eps=1e-5, moe_config=moe_cfg_16)
        self.context_d  = MoEGlobalContextModule(16, drop_rate, eps=1e-5, moe_config=moe_cfg_16)

        # Build picking heads
        # P-branch from model_PS1
        self.p_branch = PS1PickingBranch(
            in_dim=16,
            out_samples=self.in_samples,
            drop_rate=self.drop_rate,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            original_compatible=original_compatible,
            use_custom_lstm=False,  # set True if you want model_PS1's custom-lstm style
            moe_config=moe_cfg_16
        )
        # S-branch from model_PS2
        self.s_branch = PS2PickingBranch(
            in_dim=16,
            out_samples=self.in_samples,
            drop_rate=self.drop_rate,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            original_compatible=original_compatible,
            moe_config=moe_cfg_16
        )

    def forward(self, x, logits=False):
        """
        x shape: [batch, in_samples, in_channels] => as usual.
        Returns: (p_wave, s_wave), balance_loss
        """
        # 1) Reorder to [batch, channels, samples]
        x = x.permute(0, 2, 1)
        # Shared trunk
        feat = self.feature_extractor(x)
        feat, _, router_dp0 = self.context_dp0(feat)
        feat, _, router_dp1 = self.context_dp1(feat)
        feat = self.cnn_res_stack(feat)
        feat = self.bi_lstm(feat)
        feat, _, router_d0 = self.context_d0(feat)
        feat, _, router_d = self.context_d(feat)
        # feat now shape: [batch, 16, reduced_length], typically 16 x (6000/2^7= ~46 or so)

        # 2) P wave from model_PS1 branch
        p_pred = self.p_branch(feat, logits=logits)  # => [batch, 6000]

        # 3) S wave from model_PS2 branch
        s_pred = self.s_branch(feat, logits=logits)  # => [batch, 6000]

        # 4) gather MoE router logits for load-balancing
        router_list = []
        if router_dp0 is not None: router_list.append(router_dp0)
        if router_dp1 is not None: router_list.append(router_dp1)
        if router_d0  is not None: router_list.append(router_d0)
        if router_d   is not None: router_list.append(router_d)

        if router_list:
            balance_loss = torch.tensor(sum(load_balancing_loss_func(r, None) for r in router_list) / len(router_list))
        else:
            balance_loss = torch.tensor(0.0, device=x.device)

        return (p_pred, s_pred), balance_loss


# ---------------------------
#  Example usage / quick test
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMG1().to(device)

    x = torch.randn(1, 6000, 3).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    outputs, loss = model(x)
    print(outputs[0].shape)
    for i in range(1, len(outputs)):
        print(outputs[i].shape)

    # from torchview import draw_graph
    # model_graph = draw_graph(model, input_data=x, depth=2)
    # model_graph.visual_graph.render("model_structure", format="pdf")
    # print("模型图已保存为 model_structure.pdf")
