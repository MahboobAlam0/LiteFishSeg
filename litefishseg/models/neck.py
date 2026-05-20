import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DWConv, ConvBN


class BiFPNNode(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.w    = nn.Parameter(torch.ones(2))
        self.conv = DWConv(ch, ch)
        self.act  = nn.ReLU()

    def forward(self, x1, x2):
        w = self.act(self.w) / (self.act(self.w).sum() + 1e-4)
        return self.conv(w[0] * x1 + w[1] * x2)


class BiFPNLayer(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.td_p4 = BiFPNNode(ch)
        self.td_p3 = BiFPNNode(ch)
        self.bu_p4 = BiFPNNode(ch)
        self.bu_p5 = BiFPNNode(ch)

    def forward(self, feats):
        p3, p4, p5 = feats
        p4t = self.td_p4(p4, F.interpolate(p5,  size=p4.shape[2:], mode="nearest"))
        p3t = self.td_p3(p3, F.interpolate(p4t, size=p3.shape[2:], mode="nearest"))
        p4b = self.bu_p4(p4t, F.max_pool2d(p3t, 2, 2))
        p5b = self.bu_p5(p5,  F.max_pool2d(p4b, 2, 2))
        return [p3t, p4b, p5b]


class BiFPN(nn.Module):
    def __init__(self, in_channels, neck_ch=128, repeats=2):
        super().__init__()
        c3, c4, c5 = in_channels
        self.lat3 = ConvBN(c3, neck_ch, 1)
        self.lat4 = ConvBN(c4, neck_ch, 1)
        self.lat5 = ConvBN(c5, neck_ch, 1)
        self.layers = nn.ModuleList([BiFPNLayer(neck_ch) for _ in range(repeats)])
        self.out_channels = [neck_ch, neck_ch, neck_ch]

    def forward(self, feats):
        out = [self.lat3(feats[0]), self.lat4(feats[1]), self.lat5(feats[2])]
        for layer in self.layers:
            out = layer(out)
        return out
