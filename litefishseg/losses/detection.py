from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from ..config import CFG


def _giou_loss(pred, gt, eps=1e-7):
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = gt.unbind(-1)
    inter = ((torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(0) *
             (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(0))
    ap  = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ag  = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    u   = ap + ag - inter + eps
    iou = inter / u
    ec  = ((torch.max(px2, gx2) - torch.min(px1, gx1)).clamp(0) *
           (torch.max(py2, gy2) - torch.min(py1, gy1)).clamp(0)) + eps
    return 1 - (iou - (ec - u) / ec)


def _centerness(ltrb):
    l, t, r, b = ltrb.unbind(-1)
    return torch.sqrt(
        (torch.min(l, r) / (torch.max(l, r) + 1e-6)) *
        (torch.min(t, b) / (torch.max(t, b) + 1e-6))
    ).clamp(0, 1)


class DiceLoss(nn.Module):
    """Vectorised soft Dice over foreground classes (background excluded)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.s = smooth

    def forward(self, pred, target):
        ps  = F.softmax(pred[:, 1:], dim=1)
        B, C, H, W = ps.shape
        fg  = (target > 0).float()
        tgt = (target - 1).clamp(min=0)
        oh  = F.one_hot(tgt, C).permute(0, 3, 1, 2).float() * fg.unsqueeze(1)
        inter = (ps * oh).sum((2, 3))
        dice  = (2 * inter + self.s) / ((ps + oh).sum((2, 3)) + self.s)
        return 1 - dice.mean()


class LiteFishSegLoss(nn.Module):
    """FCOS spatial-radius assignment + Focal classification + GIoU box + centerness."""

    def __init__(self, num_classes, strides=None,
                 cls_w=1.0, box_w=2.0, ctr_w=1.0, seg_w=3.0, radius=1.5):
        super().__init__()
        self.nc       = num_classes
        self.strides  = strides or [8, 16, 32]
        self.cls_w    = cls_w
        self.box_w    = box_w
        self.ctr_w    = ctr_w
        self.seg_w    = seg_w
        self.radius   = float(radius)
        self.seg_ce   = nn.CrossEntropyLoss(ignore_index=255)
        self.seg_dice = DiceLoss()

    def forward(self, outputs, bboxes, labels, num_objects, seg_masks):
        det_outs = outputs["det_outputs"]
        semantic = outputs["semantic"]
        dev = semantic.device
        BS  = int(bboxes.shape[0])
        IS  = float(CFG.img_size)
        M   = int(bboxes.shape[1])

        vb  = torch.arange(M, device=dev).unsqueeze(0) < num_objects.unsqueeze(1)
        ba  = bboxes * IS
        cx, cy = ba[..., 0], ba[..., 1]
        bw, bh = ba[..., 2], ba[..., 3]
        md  = torch.max(bw, bh)

        pcl, tcl, pbx, tbx, pct, gcxl, gcyl = [], [], [], [], [], [], []
        n_pos = 0

        for lvl, out in enumerate(det_outs):
            B, C, H, W = out["cls"].shape
            s  = float(self.strides[lvl])
            if   lvl == 0: vs = md < 128.0
            elif lvl == 1: vs = (md > 32.0) & (md < 320.0)
            else:          vs = md > 128.0
            vm = vb & vs
            if not vm.any():
                continue

            yc = (torch.arange(H, device=dev).float() + 0.5) * s
            xc = (torch.arange(W, device=dev).float() + 0.5) * s
            gy, gx = torch.meshgrid(yc, xc, indexing="ij")
            gxf = gx.flatten()
            gyf = gy.flatten()

            col = gxf / s - 0.5
            row = gyf / s - 0.5
            dx  = torch.abs(col.view(1, 1, -1) - torch.floor(cx / s).unsqueeze(-1))
            dy  = torch.abs(row.view(1, 1, -1) - torch.floor(cy / s).unsqueeze(-1))
            mm  = (dx <= self.radius) & (dy <= self.radius) & vm.unsqueeze(-1)
            if not mm.any():
                continue

            area = (bw * bh).unsqueeze(-1).expand(-1, -1, H * W).clone()
            area[~mm] = float("inf")
            _, mi = area.min(dim=1)
            vp    = torch.isfinite(area.min(dim=1).values)
            bi, gi = torch.where(vp)
            if len(bi) == 0:
                continue

            mi2 = mi[bi, gi]
            cf  = out["cls"].view(BS, self.nc, H * W)
            pcl.append(cf[bi, :, gi])
            lb  = labels[bi, mi2]
            tc  = torch.zeros(len(lb), self.nc, device=dev)
            tc[torch.arange(len(lb)), lb] = 1.0
            tcl.append(tc)

            bf  = out["box"].view(BS, 4, H * W)
            lt  = bf[bi, :, gi]
            pct.append(out["ctr"].view(BS, 1, H * W)[bi, 0, gi])

            cxa = gxf[gi]
            cya = gyf[gi]
            gcxl.append(cxa)
            gcyl.append(cya)
            pbx.append(torch.stack([cxa - lt[:, 0], cya - lt[:, 1],
                                     cxa + lt[:, 2], cya + lt[:, 3]], 1))
            gcx_ = cx[bi, mi2]
            gcy_ = cy[bi, mi2]
            gbw_ = bw[bi, mi2]
            gbh_ = bh[bi, mi2]
            tbx.append(torch.stack([gcx_ - gbw_ / 2, gcy_ - gbh_ / 2,
                                     gcx_ + gbw_ / 2, gcy_ + gbh_ / 2], 1))
            n_pos += len(bi)

        if n_pos > 0:
            lc   = sigmoid_focal_loss(torch.cat(pcl), torch.cat(tcl), 0.25, 2.0, "sum")
            lb2  = _giou_loss(torch.cat(pbx), torch.cat(tbx)).sum()
            ac   = torch.cat(gcxl)
            ay   = torch.cat(gcyl)
            ta   = torch.cat(tbx)
            gl   = torch.stack([
                (ac - ta[:, 0]).clamp(0), (ay - ta[:, 1]).clamp(0),
                (ta[:, 2] - ac).clamp(0), (ta[:, 3] - ay).clamp(0),
            ], 1)
            lct  = F.binary_cross_entropy_with_logits(
                torch.cat(pct), _centerness(gl), reduction="sum")
        else:
            lc  = sum(o["cls"].sum() * 0.0 for o in det_outs)
            lb2 = sum(o["box"].sum() * 0.0 for o in det_outs)
            lct = sum(o["ctr"].sum() * 0.0 for o in det_outs)

        norm = float(max(n_pos, 1))
        mr   = F.interpolate(seg_masks.unsqueeze(1).float(),
                             size=semantic.shape[2:], mode="nearest").squeeze(1).long()
        ls   = self.seg_w * (self.seg_ce(semantic, mr) + self.seg_dice(semantic, mr))
        total = self.cls_w * lc / norm + self.box_w * lb2 / norm + self.ctr_w * lct / norm + ls
        return {"total": total,
                "cls": self.cls_w * lc / norm,
                "box": self.box_w * lb2 / norm,
                "ctr": self.ctr_w * lct / norm,
                "seg": ls}
