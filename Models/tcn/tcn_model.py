import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """Remove right-side padding to keep causality."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal pad on the left via right-pad + chomp

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)

class TCNBackbone(nn.Module):
    def __init__(self, in_features, channels=(64, 64, 64, 64), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        ch_in = in_features
        for i, ch_out in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(ch_in, ch_out, kernel_size, dilation, dropout))
            ch_in = ch_out
        self.net = nn.Sequential(*layers)

    def forward(self, x_btf):
        # x: [B,T,F] -> [B,F,T]
        x = x_btf.transpose(1, 2)
        y = self.net(x)         # [B,C,T]
        return y.transpose(1, 2)  # [B,T,C]

class MultiHeadTCN(nn.Module):
    def __init__(self, in_features, tcn_channels=(64,64,64,64), kernel_size=3, dropout=0.1,
                 num_seg_classes=3):
        super().__init__()
        self.backbone = TCNBackbone(in_features, tcn_channels, kernel_size, dropout)
        emb_dim = tcn_channels[-1]

        # Head 1: segmentation (per-timestep class logits)
        self.seg_head = nn.Linear(emb_dim, num_seg_classes)

        # Head 2: progress regression (per-timestep scalar)
        self.prog_head = nn.Linear(emb_dim, 1)

    def forward(self, x_btf):
        h = self.backbone(x_btf)                 # [B,T,C]
        seg_logits = self.seg_head(h)            # [B,T,K]
        prog = torch.sigmoid(self.prog_head(h))  # [B,T,1] in [0,1]
        return seg_logits, prog.squeeze(-1)      # prog: [B,T]
