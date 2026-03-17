import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from calflops import calculate_flops


class AFTA(nn.Module):


    def __init__(self, channels: int):
        super(AFTA, self).__init__()


        t = int(abs((math.log(channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.channel_conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.channel_sigmoid = nn.Sigmoid()


        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


        self.edge_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )


        self.fusion_weights = nn.Parameter(torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        channel_pooled = F.adaptive_avg_pool2d(x, 1)
        channel_reshaped = channel_pooled.squeeze(-1).permute(0, 2, 1)
        channel_att_raw = self.channel_conv(channel_reshaped)
        channel_att = self.channel_sigmoid(channel_att_raw).permute(0, 2, 1).unsqueeze(-1)


        spatial_att = self.spatial_gate(x)


        edge_proxy = x - self.edge_pool(x)
        edge_att = self.edge_gate(edge_proxy)


        x_channel_enhanced = x * channel_att
        x_spatial_enhanced = x * spatial_att
        x_edge_enhanced = x * edge_att


        weights = F.softmax(self.fusion_weights, dim=0)


        out = (weights[0] * x_channel_enhanced +
               weights[1] * x_spatial_enhanced +
               weights[2] * x_edge_enhanced)

        return out