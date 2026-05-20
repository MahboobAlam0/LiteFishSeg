"""
LiteFishSeg v2 — Publication-Grade Multi-Class Qualitative Results
==================================================================
Automatically samples ONE representative image per class, runs inference,
and produces a clean publication figure suitable for IEEE / Springer.

Usage:
    # Default: 12-class grid from test split
    python visualize_pub.py --weights runs/.../best.pt --data ./USISDataset

    # Custom class count / layout
    python visualize_pub.py --weights runs/.../best.pt --data ./USISDataset \
        --n-classes 16 --cols 4 --split val

    # Pick specific classes by name
    python visualize_pub.py --weights runs/.../best.pt --data ./USISDataset \
        --classes "Diver,Shark,Stingray,Octopus,Lionfish,SeaTurtle"

    # Export PDF for submission
    python visualize_pub.py --weights runs/.../best.pt --data ./USISDataset \
        --format pdf --dpi 600
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
from matplotlib.ticker import NullLocator

# ── publication rcParams ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":      "stix",
    "font.size":             8.5,
    "axes.titlesize":        8.5,
    "figure.facecolor":      "white",
    "axes.facecolor":        "black",
    "savefig.facecolor":     "white",
    "savefig.dpi":           300,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
})

# ── project imports ───────────────────────────────────────────────────────────
from main import (
    CFG,
    LiteFishSegInference,
    BRACKISH_CLASSES,
    COLORS,
    configure_dataset,
    get_val_transforms,
)
from test import make_seg_rgba   # feathered mask from test.py


# ============================================================================
# COLOUR / STYLE HELPERS
# ============================================================================

def _rgb(bgr):
    """BGR uint8 tuple → normalised (r,g,b) float."""
    return tuple(c / 255.0 for c in reversed(bgr))


# Paul Tol high-contrast palette — print-safe, colorblind-safe
_TOL = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB", "#000000",
    "#EE7733", "#0077BB", "#33BBEE", "#EE3377",
    "#CC3311", "#009988", "#BBBBBB", "#44BB99",
]

def tol_color(idx: int) -> str:
    return _TOL[idx % len(_TOL)]


# ============================================================================
# CLASS-AWARE IMAGE SAMPLER
# ============================================================================

def build_class_index(dataset) -> dict[int, list[int]]:
    """
    Scan the dataset and return {class_id: [list of dataset indices]}
    that contain at least one instance of that class.
    Progress-printed every 500 images.
    """
    print("[Sampler] Indexing dataset by class (one-time scan)…")
    index: dict[int, list[int]] = defaultdict(list)
    for i in range(len(dataset)):
        try:
            _, _, _, labels, _ = dataset.get_raw(i)
            for lb in np.unique(labels):
                index[int(lb)].append(i)
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(dataset)} scanned…")
    print(f"[Sampler] {len(index)} classes found across {len(dataset)} images.")
    return dict(index)


def sample_one_per_class(
    dataset,
    class_index: dict[int, list[int]],
    target_classes: list[int],
    seed: int = 42,
) -> list[tuple[int, int]]:
    """
    For each class in target_classes, pick the image where that class
    occupies the largest fraction of the frame (most 'representative').
    Returns [(class_id, dataset_idx), …].
    """
    rng = random.Random(seed)
    chosen = []
    for cls_id in target_classes:
        candidates = class_index.get(cls_id, [])
        if not candidates:
            continue
        # Score candidates: prefer images where the class is large
        best_idx, best_score = candidates[0], -1.0
        # Limit search to 40 candidates to avoid slow scans
        for di in rng.sample(candidates, min(40, len(candidates))):
            try:
                _, _, bboxes, labels, mask = dataset.get_raw(di)
                # Score = fraction of pixels belonging to this class
                if mask is not None and mask.size > 0:
                    score = float((mask == cls_id + 1).mean())
                elif len(bboxes) > 0:
                    areas = bboxes[:, 2] * bboxes[:, 3]  # wh
                    score = float(areas[labels == cls_id].sum()) if np.any(labels == cls_id) else 0.0
                else:
                    score = 0.0
                if score > best_score:
                    best_score, best_idx = score, di
            except Exception:
                continue
        chosen.append((cls_id, best_idx))
    return chosen


# ============================================================================
# SINGLE-CELL RENDERER
# ============================================================================

def render_cell(ax, img_rgb: np.ndarray, result: dict,
                class_name: str, cls_color_bgr: tuple,
                panel_idx: int, conf_thresh: float = 0.25):
    """
    Render one cell of the publication grid:
      - Dark background
      - Image
      - Feathered segmentation overlay (only target class shown prominently)
      - Detection boxes (only confident ones)
      - Class label badge (bottom-left)
      - Confidence badge (top-right)
    """
    h, w = img_rgb.shape[:2]
    ax.imshow(img_rgb, aspect="auto")

    # ── segmentation overlay ────────────────────────────────────────────────
    mask = result.get("mask")
    nc   = CFG.num_classes
    if mask is not None:
        rgba = make_seg_rgba(mask, nc, COLORS, alpha=160, feather_px=9)
        ax.imshow(rgba, aspect="auto")

    # ── detection boxes ─────────────────────────────────────────────────────
    boxes  = result.get("boxes")
    scores = result.get("scores")
    lbls   = result.get("labels")
    if boxes is not None and len(boxes) > 0:
        for box, sc, lb in zip(boxes, scores, lbls):
            if float(sc) < conf_thresh:
                continue
            ci  = int(lb)
            col = _rgb(COLORS[ci % len(COLORS)])
            x1, y1, x2, y2 = box
            rect = mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                boxstyle="square,pad=0",
                linewidth=1.4, edgecolor=col, facecolor="none",
                zorder=5,
            )
            ax.add_patch(rect)
            # Score chip above box
            ax.text(
                x1, y1 - 3,
                f"{BRACKISH_CLASSES.get(ci, str(ci))}  {sc:.2f}",
                fontsize=5.5, color="white", fontweight="bold",
                va="bottom", ha="left",
                bbox=dict(facecolor=col, alpha=0.82, pad=1.2,
                          edgecolor="none", boxstyle="round,pad=0.15"),
                path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
                zorder=6,
            )

    # ── class label badge (bottom-left) ──────────────────────────────────────
    badge_col = _rgb(cls_color_bgr)
    ax.text(
        0.03, 0.04, class_name,
        transform=ax.transAxes,
        fontsize=7.5, fontweight="bold", color="white",
        va="bottom", ha="left",
        bbox=dict(facecolor=badge_col, alpha=0.88,
                  boxstyle="round,pad=0.35", edgecolor="white",
                  linewidth=0.5),
        path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
        zorder=7,
    )

    # ── panel index label (top-left) ─────────────────────────────────────────
    ax.text(
        0.02, 0.97,
        f"({chr(96 + panel_idx)})",    # (a), (b), (c) …
        transform=ax.transAxes,
        fontsize=7.5, fontweight="bold",
        color="white", va="top", ha="left",
        path_effects=[pe.withStroke(linewidth=2.0, foreground="black")],
        zorder=7,
    )

    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================================
# PUBLICATION FIGURE BUILDER
# ============================================================================

def build_publication_figure(
    cells: list[dict],   # [{class_id, class_name, img_rgb, result}, …]
    n_cols: int,
    title: str,
    caption: str,
) -> plt.Figure:
    """
    Assemble the N-panel publication figure.
    cells must already be ordered as desired.
    """
    n     = len(cells)
    n_rows = (n + n_cols - 1) // n_cols

    # Cell aspect ratio: 4:3
    cell_w = 3.4          # inches per column
    cell_h = cell_w * 0.72
    fig_w  = cell_w * n_cols + 0.25
    fig_h  = cell_h * n_rows + 1.10   # extra for title + caption

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Title
    fig.text(
        0.5, 0.995,
        title,
        ha="center", va="top",
        fontsize=11.5, fontweight="bold",
        color="#0d1b2a", family="serif",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )
    # Subtitle
    fig.text(
        0.5, 0.973,
        caption,
        ha="center", va="top",
        fontsize=8.0, color="#44445a",
        fontstyle="italic", family="serif",
    )
    fig.add_artist(mlines.Line2D(
        [0.03, 0.97], [0.960, 0.960],
        transform=fig.transFigure, clip_on=False,
        color="#9098b0", linewidth=0.7,
    ))

    top    = 0.952
    bottom = 0.055
    left   = 0.010
    right  = 0.990
    h_space = 0.020
    w_space = 0.012

    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        top=top, bottom=bottom, left=left, right=right,
        hspace=h_space, wspace=w_space,
    )

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, n_cols)
        ax   = fig.add_subplot(gs[r, c])
        ax.set_facecolor("black")
        render_cell(
            ax,
            cell["img_rgb"],
            cell["result"],
            cell["class_name"],
            cell["cls_color_bgr"],
            panel_idx=idx + 1,
            conf_thresh=0.25,
        )

    # Hide empty cells
    total_cells = n_rows * n_cols
    for idx in range(n, total_cells):
        r, c = divmod(idx, n_cols)
        ax   = fig.add_subplot(gs[r, c])
        ax.axis("off")
        ax.set_facecolor("white")

    # Class legend (bottom)
    handles = [
        mpatches.Patch(
            facecolor=_rgb(COLORS[cell["class_id"] % len(COLORS)]),
            label=cell["class_name"],
            edgecolor="#444444", linewidth=0.5,
        )
        for cell in cells
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(n_cols * 2, len(cells)),
        fontsize=6.8,
        frameon=True,
        facecolor="white",
        edgecolor="#c0c8d4",
        title="Detected classes",
        title_fontsize=7.0,
        handlelength=1.0,
        handleheight=0.85,
        columnspacing=0.8,
        handletextpad=0.5,
    )

    return fig


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run(args):
    # ── load checkpoint & configure classes ──────────────────────────────────
    weights = Path(args.weights)
    if not weights.exists():
        # auto-find latest best.pt
        runs = sorted(Path("runs").glob("litefishseg_*/best.pt"))
        if not runs:
            print("[ERROR] --weights not found."); sys.exit(1)
        weights = runs[-1]
        print(f"[Auto] Using: {weights}")

    ck = torch.load(str(weights), map_location="cpu", weights_only=False)
    ck_cfg = ck.get("config", {})
    classes = ck_cfg.get("classes", CFG.classes)
    CFG.classes = classes
    CFG.num_classes = len(classes)
    BRACKISH_CLASSES.clear()
    BRACKISH_CLASSES.update({i: n for i, n in enumerate(classes)})

    # ── build inference engine ────────────────────────────────────────────────
    inf = LiteFishSegInference(
        weights_path=str(weights),
        device=args.device,
        img_size=args.img_size,
        conf=0.25,
        iou=0.30,
        preprocess=True,
    )

    # ── load dataset ──────────────────────────────────────────────────────────
    DatasetClass, _ = configure_dataset(args.data)
    tf  = get_val_transforms(args.img_size)
    ds  = DatasetClass(args.data, args.split, args.img_size, tf)

    # ── determine target classes ──────────────────────────────────────────────
    if args.classes:
        name2id = {v.lower(): k for k, v in BRACKISH_CLASSES.items()}
        target_ids = []
        for name in args.classes.split(","):
            name = name.strip()
            cid  = name2id.get(name.lower())
            if cid is None:
                print(f"[WARN] Class '{name}' not found — skipping.")
            else:
                target_ids.append(cid)
        if not target_ids:
            print("[ERROR] None of the specified classes were found."); sys.exit(1)
    else:
        # Build class index and pick n_classes most frequent classes
        class_index = build_class_index(ds)
        # Sort by frequency (most images first), then pick n_classes
        sorted_by_freq = sorted(class_index, key=lambda c: len(class_index[c]), reverse=True)
        # Exclude background (0) if present
        sorted_by_freq = [c for c in sorted_by_freq if c >= 0]
        target_ids = sorted_by_freq[:args.n_classes]
        print(f"[Classes] Selected {len(target_ids)} classes: "
              f"{[BRACKISH_CLASSES.get(c, str(c)) for c in target_ids]}")

    if not hasattr(run, '_class_index'):
        class_index = build_class_index(ds)
    samples = sample_one_per_class(ds, class_index, target_ids, seed=args.seed)

    # ── run inference and collect cells ──────────────────────────────────────
    print(f"\n[Inference] Running on {len(samples)} representative images…")
    cells = []
    for cls_id, ds_idx in samples:
        try:
            img, img_path, _, _, _ = ds.get_raw(ds_idx)
        except Exception as e:
            print(f"  [WARN] Could not load ds[{ds_idx}]: {e}"); continue

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result  = inf.predict(img_bgr)

        cls_name    = BRACKISH_CLASSES.get(cls_id, str(cls_id))
        cls_color   = COLORS[cls_id % len(COLORS)]

        print(f"  [{cls_id:3d}] {cls_name:25s}  "
              f"{len(result['boxes']) if result['boxes'] is not None else 0} det  "
              f"{result['time_ms']:.1f} ms  "
              f"({Path(img_path).name})")

        cells.append({
            "class_id":    cls_id,
            "class_name":  cls_name,
            "img_rgb":     img,       # already RGB from get_raw
            "result":      result,
            "cls_color_bgr": cls_color,
        })

    if not cells:
        print("[ERROR] No valid cells collected."); sys.exit(1)

    # ── build figure ─────────────────────────────────────────────────────────
    n_cols  = args.cols
    title   = "LiteFishSeg v2 — Qualitative Results Across Species"
    caption = (
        f"Segmentation and detection results on USIS16K test set  ·  "
        f"{len(cells)} representative classes shown  ·  "
        f"Conf ≥ 0.25  ·  img size {args.img_size}px"
    )

    print("\n[Figure] Building publication figure…")
    fig = build_publication_figure(cells, n_cols=n_cols,
                                   title=title, caption=caption)

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qualitative_results_{len(cells)}cls.{args.fmt}"
    fig.savefig(str(out_path), dpi=args.dpi,
                bbox_inches="tight", facecolor="white",
                metadata={"Creator": "LiteFishSeg v2"})
    plt.close(fig)
    print(f"\n[Done] Saved → {out_path}  ({args.dpi} dpi, {args.fmt.upper()})")

    # Auto-open
    try:
        import os; os.startfile(str(out_path))
    except AttributeError:
        try:
            import subprocess
            subprocess.run(["xdg-open", str(out_path)], check=False)
        except Exception:
            pass


# ============================================================================
# CLI
# ============================================================================

def main():
    pa = argparse.ArgumentParser(
        description="Publication-grade multi-class qualitative results — LiteFishSeg v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pa.add_argument("--weights",   default=None,
                    help="Path to checkpoint .pt (auto-finds latest if omitted)")
    pa.add_argument("--data",      default="./USISDataset",
                    help="Dataset root (must contain annotations/instances_<split>.json)")
    pa.add_argument("--split",     default="test",
                    choices=["train", "val", "test"],
                    help="Dataset split to sample from")
    pa.add_argument("--n-classes", type=int, default=12,
                    help="Number of diverse classes to show (ignored if --classes set)")
    pa.add_argument("--classes",   default=None,
                    help="Comma-separated class names to show, e.g. "
                         "'Diver,Shark,Stingray,Octopus'")
    pa.add_argument("--cols",      type=int, default=4,
                    help="Number of columns in the figure grid")
    pa.add_argument("--img-size",  type=int, default=640)
    pa.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--out-dir",   default="./pub_results")
    pa.add_argument("--format",    dest="fmt", default="png",
                    choices=["png", "pdf", "svg"],
                    help="pdf recommended for IEEE/Springer submission")
    pa.add_argument("--dpi",       type=int, default=300,
                    help="600 for camera-ready, 300 for draft")
    pa.add_argument("--seed",      type=int, default=42,
                    help="Random seed for representative image sampling")
    args = pa.parse_args()
    run(args)


if __name__ == "__main__":
    main()