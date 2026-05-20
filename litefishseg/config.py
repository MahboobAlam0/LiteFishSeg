from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./runs"


CFG = Config()

BRACKISH_CLASSES: Dict[int, str] = {i: n for i, n in enumerate(CFG.classes)}
CLASS_TO_IDX: Dict[str, int] = {n: i for i, n in BRACKISH_CLASSES.items()}

np.random.seed(42)
COLORS: List[Tuple[int, int, int]] = [
    tuple(int(x) for x in np.random.randint(50, 230, 3))
    for _ in range(max(CFG.num_classes, 20))
]
