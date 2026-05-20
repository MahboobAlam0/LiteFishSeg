import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class MobileNetV3LargeBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        f = mobilenet_v3_large(weights=w).features
        self.stage1 = nn.Sequential(*f[:7])
        self.stage2 = nn.Sequential(*f[7:13])
        self.stage3 = nn.Sequential(*f[13:])
        self.out_channels = [40, 112, 960]

    def forward(self, x):
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return [p3, p4, p5]
