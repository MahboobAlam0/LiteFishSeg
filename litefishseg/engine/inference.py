from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from ..config import CFG, BRACKISH_CLASSES, COLORS
from ..data.preprocessing import UnderwaterPreprocessor
from ..models.detector import build_model


class LiteFishSegInference:
    def __init__(self, weights_path, device="cuda", img_size=640,
                 conf=0.25, iou=0.45, preprocess=True):
        self.device   = device
        self.img_size = img_size
        self.conf     = conf
        self.iou      = iou
        self.strides  = CFG.fcos_strides

        ck = torch.load(weights_path, map_location=device)
        nc = ck["model_state_dict"]["det_head.cls_pred.weight"].shape[0]
        mc = ck.get("config", {})
        self.model = build_model(
            nc, False,
            mc.get("neck_ch",       CFG.neck_channels),
            mc.get("bifpn_repeats", CFG.bifpn_repeats),
            mc.get("mask_dim",      CFG.mask_dim),
        )
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(device).eval()
        self.pre = UnderwaterPreprocessor() if preprocess else None
        print(f"Loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M params")

    def preprocess(self, img):
        if self.pre:
            img = self.pre(img)
        h, w    = img.shape[:2]
        sc      = self.img_size / max(h, w)
        nw, nh  = int(w * sc), int(h * sc)
        pw, ph  = (self.img_size - nw) // 2, (self.img_size - nh) // 2
        canvas  = np.full((self.img_size, self.img_size, 3), 114, np.uint8)
        canvas[ph:ph + nh, pw:pw + nw] = cv2.resize(img, (nw, nh))
        t = ((canvas.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) /
             [0.229, 0.224, 0.225])
        return (torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0)
                     .float().to(self.device),
                {"sc": sc, "pw": pw, "ph": ph, "shape": (h, w)})

    def decode(self, det):
        ab, as_, al = [], [], []
        for lvl, o in enumerate(det):
            s       = self.strides[lvl]
            cl, bx  = o["cls"][0], o["box"][0]
            ct      = o["ctr"][0].sigmoid()
            _, H, W = cl.shape
            yc = (torch.arange(H, device=cl.device).float() + 0.5) * s
            xc = (torch.arange(W, device=cl.device).float() + 0.5) * s
            gy, gx = torch.meshgrid(yc, xc, indexing="ij")
            sc2     = cl.sigmoid() * ct
            ms, ml  = sc2.max(0)
            mk      = ms > self.conf
            if not mk.any():
                continue
            gxm, gym = gx[mk], gy[mk]
            l, t_, r, b = bx[0][mk], bx[1][mk], bx[2][mk], bx[3][mk]
            ab.append(torch.stack([
                (gxm - l).clamp(0, self.img_size),
                (gym - t_).clamp(0, self.img_size),
                (gxm + r).clamp(0, self.img_size),
                (gym + b).clamp(0, self.img_size),
            ], -1))
            as_.append(ms[mk])
            al.append(ml[mk])
        if not ab:
            return None, None, None
        boxes  = torch.cat(ab)
        scores = torch.cat(as_)
        lbs    = torch.cat(al)
        k      = nms(boxes, scores, self.iou)
        return boxes[k], scores[k], lbs[k]

    @torch.no_grad()
    def predict(self, img):
        t, meta = self.preprocess(img)
        t0  = time.perf_counter()
        out = self.model(t)
        ms  = (time.perf_counter() - t0) * 1000
        boxes, scores, labels = self.decode(out["det_outputs"])
        sem = out["semantic"][0].argmax(0).cpu().numpy()
        h, w = meta["shape"]
        if boxes is not None:
            boxes = boxes.cpu().numpy()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta["pw"]) / meta["sc"]
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta["ph"]) / meta["sc"]
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
        mc   = sem[meta["ph"]:meta["ph"] + int(h * meta["sc"]),
                   meta["pw"]:meta["pw"] + int(w * meta["sc"])]
        mask = cv2.resize(mc.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST)
        return {"boxes": boxes, "scores": scores, "labels": labels,
                "mask": mask, "time_ms": ms}

    def visualize(self, img, res, alpha=0.45):
        vis = img.copy()
        if res["mask"] is not None:
            ov = np.zeros_like(img)
            for ci in range(len(BRACKISH_CLASSES)):
                ov[res["mask"] == ci + 1] = COLORS[ci]
            vis = cv2.addWeighted(vis, 1 - alpha, ov, alpha, 0)
        if res["boxes"] is not None:
            for box, sc, lb in zip(res["boxes"], res["scores"], res["labels"]):
                x1, y1, x2, y2 = box.astype(int)
                col  = COLORS[int(lb) % len(COLORS)]
                name = BRACKISH_CLASSES.get(int(lb), str(lb))
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                txt  = f"{name}:{sc:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), col, -1)
                cv2.putText(vis, txt, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"{res['time_ms']:.1f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return vis
