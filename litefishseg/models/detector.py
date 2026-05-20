import torch.nn as nn

from .backbone import MobileNetV3LargeBackbone
from .neck import BiFPN
from .heads import FCOSHead, FPNSegHead


class LiteFishSeg(nn.Module):
    def __init__(self, num_classes=6, pretrained=True, neck_ch=128,
                 bifpn_repeats=2, mask_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = MobileNetV3LargeBackbone(pretrained)
        self.neck     = BiFPN(self.backbone.out_channels, neck_ch, bifpn_repeats)
        self.det_head = FCOSHead(neck_ch, num_classes, num_convs=4)
        self.seg_head = FPNSegHead(neck_ch, mask_dim, num_classes + 1, hidden=128)

        for mod in [self.neck, self.seg_head]:
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        nf = self.neck(self.backbone(x))
        so = self.seg_head(nf)
        return {"det_outputs": self.det_head(nf),
                "semantic":    so["semantic"],
                "prototypes":  so["prototypes"]}

    def count_parameters(self):
        parts  = {"backbone": self.backbone, "neck": self.neck,
                  "det_head": self.det_head, "seg_head": self.seg_head}
        counts = {k: sum(x.numel() for x in v.parameters()) for k, v in parts.items()}
        counts["total"] = sum(counts.values())
        return counts


def build_model(num_classes=6, pretrained=True, neck_ch=128,
                bifpn_repeats=2, mask_dim=64) -> LiteFishSeg:
    return LiteFishSeg(num_classes, pretrained, neck_ch, bifpn_repeats, mask_dim)
