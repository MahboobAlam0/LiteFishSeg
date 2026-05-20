# LiteFishSeg

**Lightweight Real-Time Joint Detection and Segmentation for Underwater Fish**

LiteFishSeg is a speed-optimized, single-stage model that jointly detects and segments fish in underwater video at real-time throughput. It swaps the heavyweight backbone of [FishSegDet](../FishSegDet) for MobileNetV3-Large and uses a 3-level FCOS head instead of DFL, cutting parameters by ~10× while retaining strong segmentation quality.

---

## Architecture

```
Input Image (640×640 or 512×512 for max speed)
       │
       ▼
┌─────────────────────────────┐
│  UnderwaterPreprocessor     │  White balance · CLAHE · Gamma correction
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Backbone: MobileNetV3-Large│  Torchvision pretrained (ImageNet-1K)
│  (~5.4M params)             │  Extracts C3, C4, C5 feature maps
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Neck: BiFPN  (2 repeats)   │  128-channel, 3 FPN levels P3–P5
│  Bidirectional feature      │  Strides: 8, 16, 32
│  fusion with learned weights│
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌────────────┐  ┌──────────────────┐
│  Detection │  │  Segmentation    │
│  Head      │  │  Head            │
│            │  │                  │
│  FCOS:     │  │ FPN + ASPP       │
│  cls + ctr │  │ (dilations 6,12, │
│  + LTRB    │  │  18, global avg) │
│  direct    │  │                  │
│  regression│  │ 64-d prototype   │
│            │  │ + per-instance   │
│            │  │   mask coeff     │
└────────────┘  └──────────────────┘
```

| Component | Choice | Notes |
|-----------|--------|-------|
| Backbone | MobileNetV3-Large | Torchvision pretrained on ImageNet-1K, ~5.4M params |
| Neck | BiFPN × 2 | 128 channels, 3 levels (P3–P5), learnable fusion weights |
| Detection | FCOS + Focal Loss | Direct LTRB regression + centerness branch |
| Segmentation | FPN + ASPP | ASPP dilations 6/12/18 + global pool, 64-d mask prototypes |
| Assignment | Spatial radius | FCOS radius=1.5, area-sorted priority |
| Box loss | GIoU | |
| Cls loss | Focal Loss (α=0.25, γ=2.0) | |
| Seg loss | CE + Dice (3.0×) | |

---

## FishSegDet vs LiteFishSeg

| | FishSegDet | LiteFishSeg |
|---|---|---|
| Backbone | ConvNeXtV2-Large (198M) | MobileNetV3-Large (5.4M) |
| Detection head | DFL + TAL | FCOS + Focal Loss |
| FPN levels | 4 (P3–P6) | 3 (P3–P5) |
| Neck channels | 256 | 128 |
| Mask dimension | 128 | 64 |
| Approx. params | ~250–500M | ~30–40M |
| Target use case | High-accuracy offline analysis | Real-time embedded / edge |
| Speed vs accuracy | Higher accuracy | ~4–10× faster |

---

## Key Features

- **FCOS detection** — fully convolutional, anchor-free; score = `sigmoid(cls) × centerness`, no complex label assignment at inference
- **Centerness branch** — suppresses off-centre false positives without requiring IoU-aware training targets
- **Lightweight BiFPN** — 2-repeat, 128-channel pyramid; ~4× fewer neck parameters than FishSegDet
- **AMP training** — automatic mixed precision + GradScaler on every forward/backward pass
- **EMA smoothing** — exponential moving average of weights for stable final checkpoints
- **torch.compile support** — opt-in via `--compile` for further GPU throughput gains on PyTorch ≥ 2.0
- **cudnn.benchmark + channels_last** — memory layout and kernel selection tuned for convolution-heavy inference
- **Underwater-specific preprocessing** — gray-world white balance → CLAHE → gamma LUT

---

## Training Schedule

| Phase | Epochs | Backbone | Augmentation | LR (backbone / head) |
|-------|--------|----------|--------------|----------------------|
| 1 | 15 | Frozen | Standard geometry + colour | 5e-5 / 5e-4 |
| 2 | 50 | Unfrozen | Standard | 5e-5 / 5e-4 |
| 3 | 20 | Unfrozen | Heavy (fog, motion blur, distortion) | 5e-5 / 5e-4 |

Warmup: 5 epochs cosine warm-up then cosine-annealing decay.

---

## Dataset

[USIS16K](https://github.com/LiamLian0852/USIS10K) — underwater instance segmentation, COCO format.

Expected directory layout:
```
USIS16K/
├── train/
│   └── <images>
├── val/
│   └── <images>
├── test/
│   └── <images>
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── instances_test.json
```

Class names are loaded automatically from the annotation file on startup — no manual config needed.

---

## Installation

```bash
git clone https://github.com/<your-username>/LiteFishSeg.git
cd LiteFishSeg
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0 with CUDA.

---

## Training

```bash
# Train with default config (MobileNetV3-Large, 640px, USIS dataset)
python train.py --data ./USISDataset

# Use 512px for maximum speed
python train.py --data ./USISDataset --img-size 512

# Enable torch.compile for extra GPU throughput
python train.py --data ./USISDataset --compile

# Reduce validation frequency to save time (validate every 5 epochs)
python train.py --data ./USISDataset --val-interval 5

# Resume from checkpoint
python train.py --data ./USISDataset --resume runs/exp1/last.pt
```

Checkpoints and CSV training logs are saved to `runs/<timestamp>/`.

---

## Evaluation

```bash
# Full dataset evaluation — mAP + mIoU + Dice + per-class breakdown
python test.py --weights runs/exp1/best.pt --data ./USISDataset --eval

# Save per-image overlay visualizations alongside metrics
python test.py --weights runs/exp1/best.pt --data ./USISDataset --eval --save-vis
```

Reported metrics:
- Detection: mAP@0.5, mAP@0.5:0.95, per-class AP50
- Segmentation: mIoU, Dice, pixel accuracy, per-class IoU (top 10)
- Summary dashboard plot saved to `runs/eval_summary.png`

---

## Inference

```bash
# Single image
python test.py --weights runs/exp1/best.pt --image fish.jpg

# Folder of images (overlays saved to --out-dir)
python test.py --weights runs/exp1/best.pt --folder ./images/ --out-dir ./results/
```

---

## Publication Figures

```bash
# Generate IEEE/Springer-quality result panels (one image per class)
python visualization_pub.py --weights runs/exp1/best.pt --data ./USISDataset

# Export as PDF (600 dpi, camera-ready)
python visualization_pub.py --weights runs/exp1/best.pt --data ./USISDataset --format pdf
```

Outputs colorblind-safe overlays (Paul Tol palette) with feathered mask edges on a black background.

---

## Project Structure

```
LiteFishSeg/
├── litefishseg/              # Core package
│   ├── config.py             # Config dataclass, global CFG, class maps, colours
│   ├── models/
│   │   ├── blocks.py         # ConvBN, DWConv (shared building blocks)
│   │   ├── backbone.py       # MobileNetV3LargeBackbone
│   │   ├── neck.py           # BiFPN — bidirectional multi-scale feature pyramid
│   │   ├── heads.py          # FCOSHead (cls + centerness + box), FPNSegHead
│   │   └── detector.py       # LiteFishSeg end-to-end model, build_model()
│   ├── data/
│   │   ├── preprocessing.py  # UnderwaterPreprocessor (WB → CLAHE → gamma)
│   │   ├── augmentations.py  # get_train / get_val / get_heavy transforms
│   │   ├── datasets.py       # BrackishDataset (YOLO), USISDataset (COCO)
│   │   └── loaders.py        # configure_dataset(), create_dataloaders()
│   ├── losses/
│   │   └── detection.py      # LiteFishSegLoss (FCOS + Focal + GIoU + centerness + Dice)
│   ├── engine/
│   │   └── inference.py      # LiteFishSegInference — preprocess, decode, NMS, visualize
│   └── utils/
│       └── masks.py          # generate_masks_from_bboxes()
├── train.py                  # Training entry point (phases, AMP, EMA, CSV logging)
├── visualization_pub.py      # Publication-quality result figures (IEEE / Springer)
├── requirements.txt
└── README.md
```

---

## Requirements

See [requirements.txt](requirements.txt).

Core dependencies: `torch`, `torchvision`, `albumentations`, `opencv-python`, `pycocotools`, `tqdm`, `matplotlib`.

---

## License

MIT — see [LICENSE](LICENSE).
