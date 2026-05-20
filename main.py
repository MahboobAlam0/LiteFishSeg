"""
LiteFishSeg v2 — main.py
========================
All model code, datasets, loss functions and inference live here.
train.py imports from this file. This file NEVER imports from itself.
"""

import os
import time
import math
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.ops import nms, sigmoid_focal_loss

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed.")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Dataset
    dataset_root: str = "./USISDataset"
    classes: List[str] = field(default_factory=lambda: ["fish"])
    num_classes: int = 1
    img_size: int = 640
    # Model
    backbone: str = "mobilenetv3_large"
    neck_channels: int = 128
    bifpn_repeats: int = 2
    mask_dim: int = 64
    pretrained: bool = True
    # Training
    batch_size: int = 32
    num_workers: int = 4
    phase1_epochs: int = 15
    phase2_epochs: int = 50
    phase3_epochs: int = 20
    lr_backbone: float = 5e-5
    lr_head: float = 5e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    # FCOS
    fcos_radius: float = 1.5
    fcos_strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    # Loss weights
    loss_cls: float = 1.0
    loss_box: float = 2.0
    loss_ctr: float = 1.0
    loss_seg: float = 3.0
    # Inference
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    # Preprocessing
    use_clahe: bool = True
    use_white_balance: bool = True
    use_gamma: bool = True
    gamma_value: float = 1.2
    clahe_clip: float = 2.0
    # Device / output
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./runs"


CFG = Config()

BRACKISH_CLASSES: Dict[int, str] = {i: n for i, n in enumerate(CFG.classes)}
CLASS_TO_IDX: Dict[str, int] = {n: i for i, n in BRACKISH_CLASSES.items()}

np.random.seed(42)
COLORS = [tuple(int(x) for x in np.random.randint(50, 230, 3))
          for _ in range(max(CFG.num_classes, 20))]


# ============================================================================
# PREPROCESSING
# ============================================================================

class UnderwaterPreprocessor:
    def __init__(self, use_white_balance=True, use_clahe=True,
                 use_gamma=True, clahe_clip=2.0, clahe_grid=8, gamma=1.2):
        self.use_white_balance = use_white_balance
        self.use_clahe  = use_clahe
        self.use_gamma  = use_gamma
        if use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        if use_gamma:
            inv = 1.0 / gamma
            self.lut = np.array(
                [((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.use_white_balance: img = self._wb(img)
        if self.use_clahe:         img = self._clahe(img)
        if self.use_gamma:         img = cv2.LUT(img, self.lut)
        return img

    def _wb(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        aa, ab = lab[:, :, 1].mean(), lab[:, :, 2].mean()
        ln = lab[:, :, 0] / 255.0
        lab[:, :, 1] = np.clip(lab[:, :, 1] - (aa - 128) * ln * 1.1, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] - (ab - 128) * ln * 1.1, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================================
# AUGMENTATIONS
# ============================================================================

def get_train_transforms(img_size: int = 640):
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT,
                      fill=(114, 114, 114)),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.8, 1.2),
                 rotate=(-15, 15), fill=(114, 114, 114), p=0.5),
        A.OneOf([
            A.RGBShift(r_shift_limit=(-40, -10), g_shift_limit=(-15, 5),
                       b_shift_limit=(5, 25), p=1.0),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0),
        ], p=0.7),
        A.RandomBrightnessContrast((-0.4, 0.1), (-0.2, 0.3), p=0.6),
        A.HueSaturationValue(15, 25, 25, p=0.5),
        A.OneOf([A.RandomFog(p=1.0), A.GaussNoise(p=1.0)], p=0.4),
        A.MotionBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams("yolo", label_fields=["class_labels"],
                                min_visibility=0.3))


def get_val_transforms(img_size: int = 640):
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT,
                      fill=(114, 114, 114)),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams("yolo", label_fields=["class_labels"],
                                min_visibility=0.3))


def get_heavy_transforms(img_size: int = 640):
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT,
                      fill=(114, 114, 114)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(translate_percent=(-0.15, 0.15), scale=(0.7, 1.3),
                 rotate=(-25, 25), fill=(114, 114, 114), p=0.7),
        A.RGBShift((-50, -15), (-20, 10), (10, 35), p=0.8),
        A.RandomBrightnessContrast((-0.5, 0.15), (-0.3, 0.4), p=0.8),
        A.HueSaturationValue(20, 35, 35, p=0.7),
        A.OneOf([A.RandomFog(p=1.0), A.GaussNoise(p=1.0)], p=0.5),
        A.OneOf([A.MotionBlur((5, 9), p=1.0),
                 A.GaussianBlur((5, 7), p=1.0)], p=0.3),
        A.CoarseDropout(8, img_size // 16, img_size // 16, 2, fill_value=0, p=0.3),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams("yolo", label_fields=["class_labels"],
                                min_visibility=0.2))


# ============================================================================
# DATASETS
# ============================================================================

class BrackishDataset(Dataset):
    def __init__(self, root, split="train", img_size=640, transform=None):
        self.root = Path(root); self.split = split
        self.img_size = img_size; self.transform = transform
        self.img_dir   = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.mask_dir  = self.root / "masks"  / split
        self.images = sorted(
            list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))
        print(f"[{split.upper()}] {len(self.images)} images")

    def __len__(self): return len(self.images)

    def _load_labels(self, ip, w, h):
        lp = self.label_dir / f"{ip.stem}.txt"
        if not lp.exists():
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        bboxes, labels = [], []
        for line in lp.read_text().splitlines():
            p = line.strip().split()
            if len(p) < 5: continue
            labels.append(int(p[0]))
            b = [float(x) for x in p[1:5]]
            if b[0] > 1: b[0] /= w
            if b[1] > 1: b[1] /= h
            if b[2] > 1: b[2] /= w
            if b[3] > 1: b[3] /= h
            b[2] = min(b[2], 1.0); b[3] = min(b[3], 1.0)
            if b[0] - b[2]/2 < 0: b[0] = b[2]/2
            if b[0] + b[2]/2 > 1: b[0] = 1 - b[2]/2
            if b[1] - b[3]/2 < 0: b[1] = b[3]/2
            if b[1] + b[3]/2 > 1: b[1] = 1 - b[3]/2
            bboxes.append(b)
        if bboxes:
            return np.array(bboxes, np.float32), np.array(labels, np.int64)
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)

    def _load_mask(self, ip, bboxes, labels, shape):
        mp = self.mask_dir / f"{ip.stem}.png"
        if mp.exists():
            return cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        h, w = shape; mask = np.zeros((h, w), np.uint8)
        for bbox, lbl in zip(bboxes, labels):
            cx, cy, bw, bh = bbox
            x1 = max(0, int((cx - bw/2)*w)); y1 = max(0, int((cy - bh/2)*h))
            x2 = min(w, int((cx + bw/2)*w)); y2 = min(h, int((cy + bh/2)*h))
            mask[y1:y2, x1:x2] = lbl + 1
        return mask

    def get_raw(self, idx):
        ip = self.images[idx]
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        bboxes, labels = self._load_labels(ip, w, h)
        return img, str(ip), bboxes, labels, self._load_mask(ip, bboxes, labels, (h, w))

    def __getitem__(self, idx):
        img, ip, bboxes, labels, mask = self.get_raw(idx)
        if self.transform:
            t = self.transform(image=img, mask=mask,
                               bboxes=bboxes.tolist() if len(bboxes) else [],
                               class_labels=labels.tolist() if len(labels) else [])
            img = t["image"]; mask = t["mask"]
            bboxes = np.array(t["bboxes"], np.float32)      if t["bboxes"]       else np.zeros((0, 4), np.float32)
            labels = np.array(t["class_labels"], np.int64)  if t["class_labels"] else np.zeros((0,), np.int64)
        else:
            img  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        pb = np.zeros((100, 4), np.float32); pl = np.zeros((100,), np.int64)
        n  = min(len(bboxes), 100)
        if n > 0: pb[:n] = bboxes[:n]; pl[:n] = labels[:n]
        return {"image": img, "mask": mask,
                "bboxes": torch.from_numpy(pb), "labels": torch.from_numpy(pl),
                "num_objects": torch.tensor(n), "img_path": ip}


class USISDataset(Dataset):
    def __init__(self, root, split="train", img_size=640, transform=None):
        from pycocotools.coco import COCO
        self.root = Path(root); self.split = split
        self.img_size = img_size; self.transform = transform
        self.img_dir  = self.root / split
        ann = self.root / "annotations" / f"instances_{split}.json"
        self.coco    = COCO(str(ann))
        self.img_ids = sorted(self.coco.getImgIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = [c["name"] for c in cats]
        self.cat2idx = {c["id"]: i for i, c in enumerate(cats)}
        print(f"[{split.upper()}] {len(self.img_ids)} images | {len(self.classes)} classes")

    def __len__(self): return len(self.img_ids)

    def get_raw(self, idx):
        info = self.coco.loadImgs(self.img_ids[idx])[0]
        ip   = self.img_dir / info["file_name"]
        img  = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        anns  = self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.img_ids[idx]))
        bboxes, labels = [], []
        mask = np.zeros((h, w), np.uint8)
        for ann in anns:
            if "bbox" not in ann or "category_id" not in ann: continue
            ci = self.cat2idx[ann["category_id"]]
            x, y, bw, bh = ann["bbox"]
            x1 = max(0., min(float(w), x));    y1 = max(0., min(float(h), y))
            x2 = max(0., min(float(w), x+bw)); y2 = max(0., min(float(h), y+bh))
            cbw = x2-x1; cbh = y2-y1
            if cbw <= 0 or cbh <= 0: continue
            cx = (x1+cbw/2)/w; cy = (y1+cbh/2)/h
            nw = cbw/w;         nh = cbh/h
            cx = max(1e-6, min(1-1e-6, cx)); cy = max(1e-6, min(1-1e-6, cy))
            nw = max(1e-6, min(1-1e-6, nw)); nh = max(1e-6, min(1-1e-6, nh))
            if cx+nw/2>1: nw=(1-cx)*2-1e-6
            if cx-nw/2<0: nw=cx*2-1e-6
            if cy+nh/2>1: nh=(1-cy)*2-1e-6
            if cy-nh/2<0: nh=cy*2-1e-6
            bboxes.append([cx, cy, nw, nh]); labels.append(ci)
            if ann.get("segmentation"):
                m = self.coco.annToMask(ann); mask[m > 0] = ci + 1
        ba = np.array(bboxes, np.float32) if bboxes else np.zeros((0, 4), np.float32)
        la = np.array(labels, np.int64)   if labels else np.zeros((0,), np.int64)
        return img, str(ip), ba, la, mask

    def __getitem__(self, idx):
        img, ip, bboxes, labels, mask = self.get_raw(idx)
        if self.transform:
            t = self.transform(image=img, mask=mask,
                               bboxes=bboxes.tolist() if len(bboxes) else [],
                               class_labels=labels.tolist() if len(labels) else [])
            img = t["image"]; mask = t["mask"]
            bboxes = np.array(t["bboxes"], np.float32)      if t["bboxes"]       else np.zeros((0, 4), np.float32)
            labels = np.array(t["class_labels"], np.int64)  if t["class_labels"] else np.zeros((0,), np.int64)
        else:
            img  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        pb = np.zeros((100, 4), np.float32); pl = np.zeros((100,), np.int64)
        n  = min(len(bboxes), 100)
        if n > 0: pb[:n] = bboxes[:n]; pl[:n] = labels[:n]
        return {"image": img, "mask": mask,
                "bboxes": torch.from_numpy(pb), "labels": torch.from_numpy(pl),
                "num_objects": torch.tensor(n), "img_path": ip}


def configure_dataset(root: str, batch_size: int = 32) -> Tuple[type, int]:
    import json
    root_path = Path(root)
    usis_ann  = root_path / "annotations" / "instances_train.json"
    if usis_ann.exists():
        with open(usis_ann) as f:
            ann_data = json.load(f)
        classes = [c["name"] for c in ann_data.get("categories", [])]
        if not classes:
            raise ValueError(f"No categories in {usis_ann}")
        CFG.classes = classes; CFG.num_classes = len(classes)
        global BRACKISH_CLASSES, CLASS_TO_IDX, COLORS
        BRACKISH_CLASSES.clear(); BRACKISH_CLASSES.update({i: n for i, n in enumerate(classes)})
        CLASS_TO_IDX.clear();     CLASS_TO_IDX.update({n: i for i, n in BRACKISH_CLASSES.items()})
        np.random.seed(42)
        COLORS = [tuple(int(x) for x in np.random.randint(50, 230, 3))
                  for _ in range(CFG.num_classes)]
        DatasetClass = USISDataset
        print(f"[Dataset] USIS — {CFG.num_classes} classes")
        safe_bs = min(batch_size, 32)
        if safe_bs != batch_size:
            print(f"[Memory] batch {batch_size}→{safe_bs}")
        batch_size = safe_bs; CFG.batch_size = batch_size
    else:
        DatasetClass = BrackishDataset
        print(f"[Dataset] Brackish — {CFG.num_classes} classes")
    return DatasetClass, batch_size


def create_dataloaders(root: str, img_size: int = 640,
                       batch_size: int = 32, num_workers: int = 4):
    DatasetClass, batch_size = configure_dataset(root, batch_size)
    pin     = torch.cuda.is_available()
    persist = num_workers > 0
    pf      = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        DatasetClass(root, "train", img_size, get_train_transforms(img_size)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, drop_last=True, persistent_workers=persist, prefetch_factor=pf)
    val_loader = DataLoader(
        DatasetClass(root, "val", img_size, get_val_transforms(img_size)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=persist, prefetch_factor=pf)
    test_loader = DataLoader(
        DatasetClass(root, "test", img_size, get_val_transforms(img_size)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=persist, prefetch_factor=pf)
    return train_loader, val_loader, test_loader


# ============================================================================
# MODEL BLOCKS
# ============================================================================

class ConvBN(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, k, s, (k-1)//2, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(oc, momentum=0.03, eps=1e-3)
        self.act  = nn.ReLU6(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, ic, oc, k=3, s=1):
        super().__init__()
        self.dw = ConvBN(ic, ic, k, s, g=ic)
        self.pw = ConvBN(ic, oc, 1)
    def forward(self, x): return self.pw(self.dw(x))


# ============================================================================
# BACKBONE
# ============================================================================

class MobileNetV3LargeBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w   = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        f   = mobilenet_v3_large(weights=w).features
        self.stage1 = nn.Sequential(*f[:7])
        self.stage2 = nn.Sequential(*f[7:13])
        self.stage3 = nn.Sequential(*f[13:])
        self.out_channels = [40, 112, 960]

    def forward(self, x):
        p3 = self.stage1(x); p4 = self.stage2(p3); p5 = self.stage3(p4)
        return [p3, p4, p5]


# ============================================================================
# NECK — BiFPN
# ============================================================================

class BiFPNNode(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2)); self.conv = DWConv(ch, ch); self.act = nn.ReLU()
    def forward(self, x1, x2):
        w = self.act(self.w) / (self.act(self.w).sum() + 1e-4)
        return self.conv(w[0]*x1 + w[1]*x2)

class BiFPNLayer(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.td_p4 = BiFPNNode(ch); self.td_p3 = BiFPNNode(ch)
        self.bu_p4 = BiFPNNode(ch); self.bu_p5 = BiFPNNode(ch)
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
        self.lat3 = ConvBN(c3, neck_ch, 1); self.lat4 = ConvBN(c4, neck_ch, 1)
        self.lat5 = ConvBN(c5, neck_ch, 1)
        self.layers = nn.ModuleList([BiFPNLayer(neck_ch) for _ in range(repeats)])
        self.out_channels = [neck_ch, neck_ch, neck_ch]
    def forward(self, feats):
        out = [self.lat3(feats[0]), self.lat4(feats[1]), self.lat5(feats[2])]
        for l in self.layers: out = l(out)
        return out


# ============================================================================
# DETECTION HEAD — FCOS
# ============================================================================

class FCOSHead(nn.Module):
    def __init__(self, ic, nc, num_convs=4):
        super().__init__()
        self.nc = nc
        ct = []; rt = []
        for _ in range(num_convs): ct += [DWConv(ic, ic)]; rt += [DWConv(ic, ic)]
        self.cls_tower = nn.Sequential(*ct); self.reg_tower = nn.Sequential(*rt)
        self.cls_pred  = nn.Conv2d(ic, nc, 1)
        self.reg_pred  = nn.Conv2d(ic, 4, 1)
        self.ctr_pred  = nn.Conv2d(ic, 1, 1)
        self.scales    = nn.Parameter(torch.zeros(3))
        bias = -math.log(99.0)
        nn.init.constant_(self.cls_pred.bias, bias)
        for m in [self.cls_pred, self.reg_pred, self.ctr_pred]:
            nn.init.normal_(m.weight, std=0.01)

    def forward(self, feats):
        out = []
        for i, f in enumerate(feats):
            cf = self.cls_tower(f); rf = self.reg_tower(f)
            out.append({"cls": self.cls_pred(cf),
                        "box": torch.exp(self.scales[i] + self.reg_pred(rf)).clamp(max=1000),
                        "ctr": self.ctr_pred(rf)})
        return out


# ============================================================================
# SEGMENTATION HEAD
# ============================================================================

class FPNSegHead(nn.Module):
    def __init__(self, ic, mask_dim=64, nc=7, hidden=128):
        super().__init__()
        self.fuse_p4 = ConvBN(ic, ic, 1); self.fuse_p5 = ConvBN(ic, ic, 1)
        self.fuse_cv = ConvBN(ic*3, hidden, 1)
        self.proto_net = nn.Sequential(
            ConvBN(hidden, hidden, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden, hidden, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden, mask_dim, 3))
        self.sem_net = nn.Sequential(
            ConvBN(hidden, hidden, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden, hidden//2, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBN(hidden//2, hidden//2, 3),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden//2, nc, 1))

    def forward(self, feats):
        p3, p4, p5 = feats
        p4u = F.interpolate(self.fuse_p4(p4), size=p3.shape[2:], mode="bilinear", align_corners=False)
        p5u = F.interpolate(self.fuse_p5(p5), size=p3.shape[2:], mode="bilinear", align_corners=False)
        fused = self.fuse_cv(torch.cat([p3, p4u, p5u], dim=1))
        return {"prototypes": self.proto_net(fused), "semantic": self.sem_net(fused)}


# ============================================================================
# FULL MODEL
# ============================================================================

class LiteFishSeg(nn.Module):
    def __init__(self, num_classes=6, pretrained=True, neck_ch=128,
                 bifpn_repeats=2, mask_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = MobileNetV3LargeBackbone(pretrained)
        self.neck     = BiFPN(self.backbone.out_channels, neck_ch, bifpn_repeats)
        self.det_head = FCOSHead(neck_ch, num_classes, num_convs=4)
        self.seg_head = FPNSegHead(neck_ch, mask_dim, num_classes+1, hidden=128)
        for mod in [self.neck, self.seg_head]:
            for m in mod.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None: nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        nf = self.neck(self.backbone(x))
        so = self.seg_head(nf)
        return {"det_outputs": self.det_head(nf),
                "semantic":    so["semantic"],
                "prototypes":  so["prototypes"]}

    def count_parameters(self):
        p = {"backbone": self.backbone, "neck": self.neck,
             "det_head": self.det_head, "seg_head": self.seg_head}
        c = {k: sum(x.numel() for x in v.parameters()) for k, v in p.items()}
        c["total"] = sum(c.values()); return c


def build_model(num_classes=6, pretrained=True, neck_ch=128,
                bifpn_repeats=2, mask_dim=64) -> LiteFishSeg:
    return LiteFishSeg(num_classes, pretrained, neck_ch, bifpn_repeats, mask_dim)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def _giou_loss(pred, gt, eps=1e-7):
    px1,py1,px2,py2 = pred.unbind(-1); gx1,gy1,gx2,gy2 = gt.unbind(-1)
    inter = ((torch.min(px2,gx2)-torch.max(px1,gx1)).clamp(0) *
             (torch.min(py2,gy2)-torch.max(py1,gy1)).clamp(0))
    ap = (px2-px1).clamp(0)*(py2-py1).clamp(0)
    ag = (gx2-gx1).clamp(0)*(gy2-gy1).clamp(0)
    u  = ap+ag-inter+eps; iou = inter/u
    ec = ((torch.max(px2,gx2)-torch.min(px1,gx1)).clamp(0) *
          (torch.max(py2,gy2)-torch.min(py1,gy1)).clamp(0)) + eps
    return 1-(iou-(ec-u)/ec)


def _centerness(ltrb):
    l,t,r,b = ltrb.unbind(-1)
    return torch.sqrt((torch.min(l,r)/(torch.max(l,r)+1e-6)) *
                      (torch.min(t,b)/(torch.max(t,b)+1e-6))).clamp(0,1)


class DiceLoss(nn.Module):
    """Vectorised soft Dice over foreground classes (background excluded)."""
    def __init__(self, smooth=1.0): super().__init__(); self.s = smooth
    def forward(self, pred, target):
        ps  = F.softmax(pred[:, 1:], dim=1)          # [B,C,H,W] skip bg
        B,C,H,W = ps.shape
        fg  = (target > 0).float()
        tgt = (target-1).clamp(min=0)                 # remap 1..C → 0..C-1
        oh  = F.one_hot(tgt, C).permute(0,3,1,2).float() * fg.unsqueeze(1)
        inter = (ps*oh).sum((2,3))
        dice  = (2*inter+self.s)/((ps+oh).sum((2,3))+self.s)
        return 1-dice.mean()


class LiteFishSegLoss(nn.Module):
    def __init__(self, num_classes, strides=None, cls_w=1.0, box_w=2.0,
                 ctr_w=1.0, seg_w=3.0, radius=1.5):
        super().__init__()
        self.nc=num_classes; self.strides=strides or [8,16,32]
        self.cls_w=cls_w; self.box_w=box_w; self.ctr_w=ctr_w; self.seg_w=seg_w
        self.radius=float(radius)
        self.seg_ce=nn.CrossEntropyLoss(ignore_index=255); self.seg_dice=DiceLoss()

    def forward(self, outputs, bboxes, labels, num_objects, seg_masks):
        det_outs=outputs["det_outputs"]; semantic=outputs["semantic"]
        dev=semantic.device; BS=int(bboxes.shape[0]); IS=float(CFG.img_size)
        M=int(bboxes.shape[1])
        vb  = torch.arange(M,device=dev).unsqueeze(0) < num_objects.unsqueeze(1)
        ba  = bboxes*IS
        cx=ba[...,0]; cy=ba[...,1]; bw=ba[...,2]; bh=ba[...,3]
        md  = torch.max(bw,bh)
        pcl=[]; tcl=[]; pbx=[]; tbx=[]; pct=[]; gcxl=[]; gcyl=[]; n_pos=0

        for lvl,out in enumerate(det_outs):
            B,C,H,W = out["cls"].shape; s=float(self.strides[lvl])
            if   lvl==0: vs=md<128.
            elif lvl==1: vs=(md>32.)&(md<320.)
            else:        vs=md>128.
            vm=vb&vs
            if not vm.any(): continue
            yc=(torch.arange(H,device=dev).float()+.5)*s
            xc=(torch.arange(W,device=dev).float()+.5)*s
            gy,gx=torch.meshgrid(yc,xc,indexing="ij")
            gxf=gx.flatten(); gyf=gy.flatten()
            col=gxf/s - 0.5; row=gyf/s - 0.5
            dx=torch.abs(col.view(1,1,-1) - torch.floor(cx/s).unsqueeze(-1))  # [BS,M,H*W]
            dy=torch.abs(row.view(1,1,-1) - torch.floor(cy/s).unsqueeze(-1))  # [BS,M,H*W]
            mm=(dx<=self.radius)&(dy<=self.radius)&vm.unsqueeze(-1)
            if not mm.any(): continue
            area=(bw*bh).unsqueeze(-1).expand(-1,-1,H*W).clone()
            area[~mm]=float("inf")
            _,mi=area.min(dim=1); vp=torch.isfinite(area.min(dim=1).values)
            bi,gi=torch.where(vp)
            if len(bi)==0: continue
            mi2=mi[bi,gi]
            cf=out["cls"].view(BS,self.nc,H*W); pcl.append(cf[bi,:,gi])
            lb=labels[bi,mi2]; tc=torch.zeros(len(lb),self.nc,device=dev)
            tc[torch.arange(len(lb)),lb]=1.; tcl.append(tc)
            bf=out["box"].view(BS,4,H*W); lt=bf[bi,:,gi]
            pct.append(out["ctr"].view(BS,1,H*W)[bi,0,gi])
            cxa=gxf[gi]; cya=gyf[gi]; gcxl.append(cxa); gcyl.append(cya)
            pbx.append(torch.stack([cxa-lt[:,0],cya-lt[:,1],cxa+lt[:,2],cya+lt[:,3]],1))
            gcx_=cx[bi,mi2]; gcy_=cy[bi,mi2]; gbw_=bw[bi,mi2]; gbh_=bh[bi,mi2]
            tbx.append(torch.stack([gcx_-gbw_/2,gcy_-gbh_/2,gcx_+gbw_/2,gcy_+gbh_/2],1))
            n_pos+=len(bi)

        if n_pos>0:
            lc=sigmoid_focal_loss(torch.cat(pcl),torch.cat(tcl),0.25,2.0,"sum")
            lb2=_giou_loss(torch.cat(pbx),torch.cat(tbx)).sum()
            ac=torch.cat(gcxl); ay=torch.cat(gcyl); ta=torch.cat(tbx)
            gl=torch.stack([(ac-ta[:,0]).clamp(0),(ay-ta[:,1]).clamp(0),
                            (ta[:,2]-ac).clamp(0),(ta[:,3]-ay).clamp(0)],1)
            lct=F.binary_cross_entropy_with_logits(torch.cat(pct), _centerness(gl), reduction="sum")
        else:
            lc =sum(o["cls"].sum()*0. for o in det_outs)
            lb2=sum(o["box"].sum()*0. for o in det_outs)
            lct=sum(o["ctr"].sum()*0. for o in det_outs)

        norm=float(max(n_pos,1))
        mr=F.interpolate(seg_masks.unsqueeze(1).float(),
                         size=semantic.shape[2:],mode="nearest").squeeze(1).long()
        ls=self.seg_w*(self.seg_ce(semantic,mr)+self.seg_dice(semantic,mr))
        total=self.cls_w*lc/norm+self.box_w*lb2/norm+self.ctr_w*lct/norm+ls
        return {"total":total,"cls":self.cls_w*lc/norm,"box":self.box_w*lb2/norm,
                "ctr":self.ctr_w*lct/norm,"seg":ls}


# ============================================================================
# INFERENCE
# ============================================================================

class LiteFishSegInference:
    def __init__(self, weights_path, device="cuda", img_size=640,
                 conf=0.25, iou=0.45, preprocess=True):
        self.device=device; self.img_size=img_size
        self.conf=conf; self.iou=iou; self.strides=CFG.fcos_strides
        ck=torch.load(weights_path,map_location=device)
        nc=ck["model_state_dict"]["det_head.cls_pred.weight"].shape[0]
        mc=ck.get("config",{})
        self.model=build_model(nc,False,mc.get("neck_ch",CFG.neck_channels),
                               mc.get("bifpn_repeats",CFG.bifpn_repeats),
                               mc.get("mask_dim",CFG.mask_dim))
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(device).eval()
        self.pre=UnderwaterPreprocessor() if preprocess else None
        print(f"Loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M params")

    def preprocess(self,img):
        if self.pre: img=self.pre(img)
        h,w=img.shape[:2]; sc=self.img_size/max(h,w)
        nw,nh=int(w*sc),int(h*sc); pw,ph=(self.img_size-nw)//2,(self.img_size-nh)//2
        cv=np.full((self.img_size,self.img_size,3),114,np.uint8)
        cv[ph:ph+nh,pw:pw+nw]=cv2.resize(img,(nw,nh))
        t=((cv.astype(np.float32)/255.-[.485,.456,.406])/[.229,.224,.225])
        return torch.from_numpy(t.transpose(2,0,1)).unsqueeze(0).float().to(self.device),\
               {"sc":sc,"pw":pw,"ph":ph,"shape":(h,w)}

    def decode(self,det):
        ab=[]; as_=[]; al=[]
        for lvl,o in enumerate(det):
            s=self.strides[lvl]; cl=o["cls"][0]; bx=o["box"][0]; ct=o["ctr"][0].sigmoid()
            _,H,W=cl.shape
            yc=(torch.arange(H,device=cl.device).float()+.5)*s
            xc=(torch.arange(W,device=cl.device).float()+.5)*s
            gy,gx=torch.meshgrid(yc,xc,indexing="ij")
            sc=cl.sigmoid()*ct; ms,ml=sc.max(0); mk=ms>self.conf
            if not mk.any(): continue
            gxm,gym=gx[mk],gy[mk]; l,t,r,b=bx[0][mk],bx[1][mk],bx[2][mk],bx[3][mk]
            ab.append(torch.stack([(gxm-l).clamp(0,self.img_size),(gym-t).clamp(0,self.img_size),
                                   (gxm+r).clamp(0,self.img_size),(gym+b).clamp(0,self.img_size)],-1))
            as_.append(ms[mk]); al.append(ml[mk])
        if not ab: return None,None,None
        boxes=torch.cat(ab); scores=torch.cat(as_); labels=torch.cat(al)
        k=nms(boxes,scores,self.iou)
        return boxes[k],scores[k],labels[k]

    @torch.no_grad()
    def predict(self,img):
        t,meta=self.preprocess(img); t0=time.perf_counter()
        out=self.model(t); ms=(time.perf_counter()-t0)*1000
        boxes,scores,labels=self.decode(out["det_outputs"])
        sem=out["semantic"][0].argmax(0).cpu().numpy()
        h,w=meta["shape"]
        if boxes is not None:
            boxes=boxes.cpu().numpy()
            boxes[:,[0,2]]=(boxes[:,[0,2]]-meta["pw"])/meta["sc"]
            boxes[:,[1,3]]=(boxes[:,[1,3]]-meta["ph"])/meta["sc"]
            boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w)
            boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h)
            scores=scores.cpu().numpy(); labels=labels.cpu().numpy()
        mc=sem[meta["ph"]:meta["ph"]+int(h*meta["sc"]),
               meta["pw"]:meta["pw"]+int(w*meta["sc"])]
        mask=cv2.resize(mc.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
        return {"boxes":boxes,"scores":scores,"labels":labels,"mask":mask,"time_ms":ms}

    def visualize(self,img,res,alpha=0.45):
        vis=img.copy()
        if res["mask"] is not None:
            ov=np.zeros_like(img)
            for ci in range(len(BRACKISH_CLASSES)):
                ov[res["mask"]==ci+1]=COLORS[ci]
            vis=cv2.addWeighted(vis,1-alpha,ov,alpha,0)
        if res["boxes"] is not None:
            for box,sc,lb in zip(res["boxes"],res["scores"],res["labels"]):
                x1,y1,x2,y2=box.astype(int); col=COLORS[int(lb)%len(COLORS)]
                name=BRACKISH_CLASSES.get(int(lb),str(lb))
                cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
                txt=f"{name}:{sc:.2f}"; (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
                cv2.rectangle(vis,(x1,y1-th-8),(x1+tw+4,y1),col,-1)
                cv2.putText(vis,txt,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(vis,f"{res['time_ms']:.1f}ms",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
        return vis


# ============================================================================
# UTILITIES
# ============================================================================

def generate_masks_from_bboxes(data_root, split="train", method="ellipse"):
    from tqdm import tqdm
    root=Path(data_root); img_dir=root/"images"/split
    label_dir=root/"labels"/split; mask_dir=root/"masks"/split
    mask_dir.mkdir(parents=True,exist_ok=True)
    files=list(img_dir.glob("*.jpg"))+list(img_dir.glob("*.png"))
    for ip in tqdm(files,desc=f"Masks [{split}]"):
        img=cv2.imread(str(ip)); h,w=img.shape[:2]
        mask=np.zeros((h,w),np.uint8); lp=label_dir/f"{ip.stem}.txt"
        if lp.exists():
            for line in lp.read_text().splitlines():
                p=line.strip().split()
                if len(p)<5: continue
                ci=int(p[0]); cx_,cy_,bw_,bh_=map(float,p[1:5])
                if method=="ellipse":
                    cv2.ellipse(mask,(int(cx_*w),int(cy_*h)),
                                (int(bw_*w*.85/2),int(bh_*h*.85/2)),0,0,360,ci+1,-1)
                else:
                    x1=max(0,int((cx_-bw_/2)*w)); y1=max(0,int((cy_-bh_/2)*h))
                    x2=min(w,int((cx_+bw_/2)*w)); y2=min(h,int((cy_+bh_/2)*h))
                    mask[y1:y2,x1:x2]=ci+1
        cv2.imwrite(str(mask_dir/f"{ip.stem}.png"),mask)
    print(f"Generated {len(files)} masks → {mask_dir}")


# ============================================================================
# SMOKE TEST  (only runs when executing main.py directly)
# ============================================================================

if __name__ == "__main__":
    print("LiteFishSeg v2 — Smoke Test"); print("="*50)
    m=build_model(num_classes=7,pretrained=False)
    for k,v in m.count_parameters().items(): print(f"  {k:12s}: {v/1e6:.2f}M")
    x=torch.randn(2,3,640,640)
    with torch.no_grad(): o=m(x)
    for i,d in enumerate(o["det_outputs"]):
        print(f"  Level {i}: cls{tuple(d['cls'].shape)}")
    print(f"  Semantic: {tuple(o['semantic'].shape)}")
    lf=LiteFishSegLoss(7,CFG.fcos_strides,1.5)
    bx=torch.rand(2,10,4)*.5+.25; bx[...,2:]=bx[...,2:].clamp(max=.3)
    ls=lf(o,bx,torch.randint(0,7,(2,10)),torch.tensor([5,3]),torch.randint(0,8,(2,640,640)))
    for k,v in ls.items(): print(f"  {k}: {v.item():.4f}")