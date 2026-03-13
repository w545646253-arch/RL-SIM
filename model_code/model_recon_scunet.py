# -*- coding: utf-8 -*-
"""
轻量 SCUNet（6→1 通道），GroupNorm 适配小批量补丁训练。
默认 base_ch=64；可在 config 中调 CFG.RECON_BASE_CH。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def _norm(ch: int) -> nn.Module:
    for g in (16, 8, 4, 2):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.Identity()

def conv_gn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        _norm(out_ch),
        nn.ReLU(inplace=True)
    )

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv_gn_relu(in_ch, out_ch)
        self.conv2 = conv_gn_relu(out_ch, out_ch)
        self.down  = nn.Conv2d(out_ch, out_ch, 3, 2, 1)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x)
        d = self.down(x)
        return x, d

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = conv_gn_relu(in_ch + skip_ch, out_ch)
        self.conv2 = conv_gn_relu(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x); x = self.conv2(x)
        return x

class SCUNetRecon(nn.Module):
    def __init__(self, base_ch=64, in_ch=6, out_ch=1):
        super().__init__()
        c = base_ch
        self.e1 = DownBlock(in_ch, c)
        self.e2 = DownBlock(c,   c*2)
        self.e3 = DownBlock(c*2, c*4)
        self.e4 = DownBlock(c*4, c*8)
        self.bn1 = conv_gn_relu(c*8, c*8)
        self.bn2 = conv_gn_relu(c*8, c*8)
        self.u3 = UpBlock(c*8, c*4, c*4)
        self.u2 = UpBlock(c*4, c*2, c*2)
        self.u1 = UpBlock(c*2, c,   c)
        self.head = conv_gn_relu(in_ch, c)
        self.tail = nn.Conv2d(c, out_ch, 1, 1, 0)

    def forward(self, x):
        h0 = self.head(x)
        h1, d1 = self.e1(x)
        h2, d2 = self.e2(d1)
        h3, d3 = self.e3(d2)
        h4, d4 = self.e4(d3)
        b = self.bn1(d4); b = self.bn2(b)
        u3 = self.u3(b,  h3)
        u2 = self.u2(u3, h2)
        u1 = self.u1(u2, h1)
        u0 = F.interpolate(u1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        y  = self.tail(u0 + h0)
        return y
