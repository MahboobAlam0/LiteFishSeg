import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, k, s, (k - 1) // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(oc, momentum=0.03, eps=1e-3)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, ic, oc, k=3, s=1):
        super().__init__()
        self.dw = ConvBN(ic, ic, k, s, g=ic)
        self.pw = ConvBN(ic, oc, 1)

    def forward(self, x):
        return self.pw(self.dw(x))
