import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange




class SFIF(nn.Module):


    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.qkv_s = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.attn_s_dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.proj_s = nn.Conv2d(dim, dim, 1, bias=False)
        self.scale_s = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.freq_filter = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False)
        )
        self.proj_f = nn.Conv2d(dim, dim, 1, bias=False)

        self.gate_s_to_f = nn.Conv2d(dim, dim, 1)  # Spatial features gate Frequency
        self.gate_f_to_s = nn.Conv2d(dim, dim, 1)  # Frequency features gate Spatial

    def forward(self, x):
        b, c, h, w = x.shape

        q_s, k_s, v_s = self.qkv_s(x).chunk(3, dim=1)
        v_s_conv = self.attn_s_dwconv(v_s)

        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)


        attn_s = F.softmax((q_s @ k_s.transpose(-2, -1)) * self.scale_s, dim=-1)
        out_s = rearrange(attn_s @ v_s, 'b head c (h w) -> b (head c) h w', h=h, w=w)
        out_s = self.proj_s(out_s + v_s_conv)


        xf = torch.fft.rfft2(x.float(), norm='ortho')

        spatial_weights = self.freq_filter(x)
        freq_weights = torch.fft.rfft2(spatial_weights.float(), norm='ortho')


        filtered_xf = xf * freq_weights

        out_f = torch.fft.irfft2(filtered_xf, s=(h, w), norm='ortho')
        out_f = self.proj_f(out_f)


        gate_f = torch.sigmoid(self.gate_s_to_f(out_s))
        gate_s = torch.sigmoid(self.gate_f_to_s(out_f))

        fused_out = (out_s * gate_s) + (out_f * gate_f)

        return x + fused_out