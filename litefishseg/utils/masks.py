from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def generate_masks_from_bboxes(data_root, split="train", method="ellipse"):
    root      = Path(data_root)
    img_dir   = root / "images" / split
    label_dir = root / "labels" / split
    mask_dir  = root / "masks"  / split
    mask_dir.mkdir(parents=True, exist_ok=True)
    files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    for ip in tqdm(files, desc=f"Masks [{split}]"):
        img  = cv2.imread(str(ip))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        lp   = label_dir / f"{ip.stem}.txt"
        if lp.exists():
            for line in lp.read_text().splitlines():
                p = line.strip().split()
                if len(p) < 5:
                    continue
                ci   = int(p[0])
                cx_, cy_, bw_, bh_ = map(float, p[1:5])
                if method == "ellipse":
                    cv2.ellipse(mask,
                                (int(cx_ * w), int(cy_ * h)),
                                (int(bw_ * w * 0.85 / 2), int(bh_ * h * 0.85 / 2)),
                                0, 0, 360, ci + 1, -1)
                else:
                    x1 = max(0, int((cx_ - bw_ / 2) * w))
                    y1 = max(0, int((cy_ - bh_ / 2) * h))
                    x2 = min(w, int((cx_ + bw_ / 2) * w))
                    y2 = min(h, int((cy_ + bh_ / 2) * h))
                    mask[y1:y2, x1:x2] = ci + 1
        cv2.imwrite(str(mask_dir / f"{ip.stem}.png"), mask)
    print(f"Generated {len(files)} masks → {mask_dir}")
