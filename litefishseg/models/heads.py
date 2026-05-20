import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBN, DWConv


class FCOSHead(nn.Module):
    """
    Anchor-free FCOS detection head.
    Score at inference = sigmoid(cls) * sigmoid(centerness)
    """
    def __init__(self, ic, nc, num_convs=4):
        super().__init__()
        self.nc = nc
        ct, rt = [], []
        for _ in range(num_convs):
            ct.append(DWConv(ic, ic))
            rt.append(DWConv(ic, ic))
        self.cls_tower = nn.Sequential(*ct)
        self.reg_tower = nn.Sequential(*rt)
        self.cls_pred  = nn.Conv2d(ic, nc, 1)
        self.reg_pred  = nn.Conv2d(ic, 4, 1)
        self.ctr_pred  = nn.Conv2d(ic, 1, 1)
        self.scales    = nn.Parameter(torch.zeros(3))
        nn.init.constant_(self.cls_pred.bias, -math.log(99.0))
        for m in [self.cls_pred, self.reg_pred, self.ctr_pred]:
            nn.init.normal_(m.weight, std=0.01)

    def forward(self, feats):
        out = []
        for i, f in enumerate(feats):
            cf = self.cls_tower(f)
            rf = self.reg_tower(f)
            out.append({
                "cls": self.cls_pred(cf),
                "box": torch.exp(self.scales[i] + self.reg_pred(rf)).clamp(max=1000),
                "ctr": self.ctr_pred(rf),
            })
        return out


class FPNSegHead(nn.Module):
    def __init__(self, ic, mask_dim=64, nc=7, hidden=128):
        super().__init__()
        self.fuse_p4 = ConvBN(ic, ic, 1)
        self.fuse_p5 = ConvBN(ic, ic, 1)
        self.fuse_cv = ConvBN(ic * 3, hidden, 1)
        self.proto_net = nn.Sequential(
            ConvBN(hidden, hidden, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden, hidden, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden, mask_dim, 3))
        self.sem_net = nn.Sequential(
            ConvBN(hidden, hidden, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden, hidden // 2, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden // 2, hidden // 2, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden // 2, nc, 1))

    def forward(self, feats):
        p3, p4, p5 = feats
        p4u = F.interpolate(self.fuse_p4(p4), size=p3.shape[2:],
                            mode="bilinear", align_corners=False)
        p5u = F.interpolate(self.fuse_p5(p5), size=p3.shape[2:],
                            mode="bilinear", align_corners=False)
        fused = self.fuse_cv(torch.cat([p3, p4u, p5u], dim=1))
        return {"prototypes": self.proto_net(fused), "semantic": self.sem_net(fused)}
