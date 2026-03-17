import os, sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
sys.path.append(project_root)

import warnings

warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
import torch.nn.functional as F


from engine.extre_module.ultralytics_nn.conv import Conv


class DCFSB(nn.Module):

    def __init__(self, inc, dim, reduction=8, shift_size=1, eps=1e-5):
        super(DCFSB, self).__init__()

        self.height = len(inc)

        if self.height < 4:
            print(
                f"Warning: DCFSB's shift fusion is designed for >= 4 inputs, but got {self.height}. The shift operation will be simplified.")
        d = max(int(dim / reduction), 4)
        self.shift_size = shift_size
        self.eps = eps


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * self.height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

        self.conv1x1 = nn.ModuleList([])
        for i in inc:
            if i != dim:
                self.conv1x1.append(Conv(i, dim, 1))
            else:
                self.conv1x1.append(nn.Identity())

    def forward(self, in_feats_):
        in_feats = []
        for idx, layer in enumerate(self.conv1x1):
            in_feats.append(layer(in_feats_[idx]))

        B, C, H, W = in_feats[0].shape

        in_feats_stacked = torch.stack(in_feats, dim=1)


        feats_sum = torch.sum(in_feats_stacked, dim=1)


        spatial_mean = feats_sum.mean(dim=[2, 3], keepdim=True)

        squared_deviation = (feats_sum - spatial_mean).pow(2)


        contrast_weights = squared_deviation / (squared_deviation.mean(dim=[2, 3], keepdim=True) + self.eps)
        contrast_enhanced_feats = feats_sum + (feats_sum * contrast_weights)


        attn = self.mlp(self.avg_pool(contrast_enhanced_feats))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        weighted_feats = in_feats_stacked * attn


        shifted_feats = []
        for i in range(self.height):
            feat = weighted_feats[:, i, ...]
            if i % 4 == 0:
                shifted_feats.append(torch.roll(feat, shifts=self.shift_size, dims=2))
            elif i % 4 == 1:
                shifted_feats.append(torch.roll(feat, shifts=-self.shift_size, dims=2))
            elif i % 4 == 2:
                shifted_feats.append(torch.roll(feat, shifts=self.shift_size, dims=3))
            else:
                shifted_feats.append(torch.roll(feat, shifts=-self.shift_size, dims=3))

        shifted_feats_stacked = torch.stack(shifted_feats, dim=1)


        out, _ = torch.max(shifted_feats_stacked, dim=1)

        return out