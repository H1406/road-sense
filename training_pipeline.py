#   pip install ultralytics pycocotools opencv-python matplotlib seaborn
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

from __future__ import annotations
import collections
import json
import random
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


try:
    import torch
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(f"Missing library: {e}\nRun: pip install ultralytics torch")

try:
    from pycocotools import mask as coco_mask
    import cv2
except ImportError as e:
    sys.exit(f"Missing library: {e}\nRun: pip install pycocotools opencv-python")


BASE_DIR         = Path(r"d:\Khue\Theme 1")
RUBBISH_DIR      = BASE_DIR / "dataset" / "rubbish" / "train"
NOTRUBBISH_DIR   = BASE_DIR / "dataset" / "notrubbish" / "train"
RUNS_DIR         = BASE_DIR / "runs"

CLEANED_JSON        = RUBBISH_DIR / "_annotations_clean.coco.json"
YOLO_SEG_LABELS_DIR = RUBBISH_DIR / "labels_yolo_seg_12cls"
YOLO_SEG_SPLIT_DIR  = BASE_DIR / "dataset" / "yolo_seg_split_12cls"
DATASET_SEG_YAML    = BASE_DIR / "dataset" / "dataset_seg_12cls.yaml"
CLS_SPLIT_DIR       = BASE_DIR / "dataset" / "cls_split"

# ── Class reduction ─────────────────────────────────────────────────────────
DROP_CLASSES = {"Clothes", "Suitcase", "Litter"}
MERGE_MAP    = {
    "Wooden_trash": "Wooden_crate",  # merge into Wooden_crate
    "Plastic_bag" : "Garbage_bag",   # merge into Garbage_bag
}

# ── Oversample ────────────────────────────────────────────────────────────────
MAX_NOTRUBBISH   = 400     # number of not-rubbish images added as background
RARE_THRESHOLD   = 300     # classes with < 300 annotations → oversample
OVERSAMPLE_MAX   = 6       # maximum copy factor ×6

# ── Hyperparams ───────────────────────────────────────────────────────────────
VAL_RATIO = 0.2
SEED      = 42
DEVICE    = 0              # GPU 0; change to "cpu" if no GPU available

# Classifier
CLS_MODEL  = "yolo11l-cls.pt"
CLS_IMGSZ  = 224
CLS_BATCH  = 32
CLS_EPOCHS = 60
CLS_RUN    = "rubbish_cls"

# Segmentation
SEG_MODEL  = "yolo11l-seg.pt"
SEG_IMGSZ  = 640
SEG_BATCH  = 8             # increase to 16 if VRAM allows (>16 GB)
SEG_EPOCHS = 100
SEG_RUN    = "rubbish_yolo11l_seg_12cls"

# =============================================================================
# SECTION 1 – DATASET ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 1: DATASET ANALYSIS")
print("=" * 60)

RUBBISH_JSON    = RUBBISH_DIR / "_annotations.coco.json"
NOTRUBBISH_JSON = NOTRUBBISH_DIR / "_annotations.coco.json"

with open(RUBBISH_JSON) as f:
    rubbish_coco = json.load(f)
with open(NOTRUBBISH_JSON) as f:
    notrubbish_coco = json.load(f)

n_rubbish_imgs = len(rubbish_coco["images"])
n_notrubbish   = len(notrubbish_coco["images"])
n_annotations  = len(rubbish_coco["annotations"])
n_classes      = len(rubbish_coco["categories"])

print(f"  Rubbish images    : {n_rubbish_imgs:,}")
print(f"  Not-rubbish images: {n_notrubbish:,}")
print(f"  Total images      : {n_rubbish_imgs + n_notrubbish:,}")
print(f"  Total annotations : {n_annotations:,}")
print(f"  Original classes  : {n_classes}")

cat_map = {c["id"]: c["name"] for c in rubbish_coco["categories"]}
cat_map.pop(0, None)  # remove 'objects' (id=0) – generic catch-all

ann_per_class = collections.Counter(
    a["category_id"] for a in rubbish_coco["annotations"]
    if a["category_id"] != 0
)
print("\nAnnotations per class (descending):")
for cid, cnt in sorted(ann_per_class.items(), key=lambda x: -x[1]):
    print(f"  {cat_map.get(cid, cid):30s}: {cnt:6,d}")

# ── Class distribution chart ────────────────────────────────────────────────
labels = [cat_map[cid] for cid, _ in sorted(ann_per_class.items(), key=lambda x: -x[1])]
counts = [cnt          for _,  cnt in sorted(ann_per_class.items(), key=lambda x: -x[1])]

fig, ax = plt.subplots(figsize=(14, 5))
colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
bars = ax.bar(labels, counts, color=colors)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_title("Class Distribution – Rubbish Dataset")
ax.set_ylabel("Annotation count")
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            str(cnt), ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(str(BASE_DIR / "class_distribution.png"), dpi=150)
plt.close()
print("\n  Saved: class_distribution.png")

# ── Image size information ───────────────────────────────────────────────────
widths  = [i["width"]  for i in rubbish_coco["images"] if "width"  in i]
heights = [i["height"] for i in rubbish_coco["images"] if "height" in i]
print(f"\n  Width  : {min(widths)} – {max(widths)} px")
print(f"  Height : {min(heights)} – {max(heights)} px")

# =============================================================================
# SECTION 2 – DATA CLEANING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: DATA CLEANING")
print("=" * 60)

def fix_bbox_types(coco_data: dict) -> dict:
    """Coerce all bbox coordinates to float (some are stored as strings)."""
    fixed = 0
    for ann in coco_data["annotations"]:
        clean = []
        for v in ann["bbox"]:
            try:
                clean.append(float(v))
            except (ValueError, TypeError):
                clean.append(0.0)
                fixed += 1
        ann["bbox"] = clean
    print(f"  Fixed {fixed} malformed bbox values.")
    return coco_data

rubbish_coco = fix_bbox_types(rubbish_coco)

# Remove annotations with zero-area bbox
before = len(rubbish_coco["annotations"])
rubbish_coco["annotations"] = [
    a for a in rubbish_coco["annotations"]
    if a["bbox"][2] > 0 and a["bbox"][3] > 0
]
print(f"  Removed {before - len(rubbish_coco['annotations'])} zero-area annotations.")

with open(CLEANED_JSON, "w") as f:
    json.dump(rubbish_coco, f)
print(f"  Saved cleaned JSON → {CLEANED_JSON}")

# =============================================================================
# SECTION 3 – CLASS REDUCTION: DROP AND MERGE
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: CLASS REDUCTION (DROP + MERGE) → 12 CLASSES")
print("=" * 60)
print(f"  Drop : {DROP_CLASSES}")
print(f"  Merge: {MERGE_MAP}")

skip_names = DROP_CLASSES | set(MERGE_MAP.keys())
keep_cats  = [
    c for c in rubbish_coco["categories"]
    if c["id"] != 0 and c["name"] not in skip_names
]
name_to_yolo = {c["name"]: idx for idx, c in enumerate(keep_cats)}
CLASS_NAMES  = [c["name"] for c in keep_cats]   # 12 classes

# Map merge sources to the same YOLO id as their target
for src, dst in MERGE_MAP.items():
    name_to_yolo[src] = name_to_yolo[dst]

coco_name     = {c["id"]: c["name"] for c in rubbish_coco["categories"]}
new_cat_remap = {
    c["id"]: name_to_yolo[coco_name[c["id"]]]
    for c in rubbish_coco["categories"]
    if c["id"] != 0 and coco_name[c["id"]] not in DROP_CLASSES
}

print(f"\n  Classes after reduction: {len(CLASS_NAMES)}")
print(f"  Classes: {CLASS_NAMES}")

print("\n  COCO id → YOLO id:")
for c in rubbish_coco["categories"]:
    name = coco_name[c["id"]]
    if c["id"] == 0:
        print(f"    {c['id']:2d}  {name:25s}  → DROPPED (id=0)")
    elif name in DROP_CLASSES:
        print(f"    {c['id']:2d}  {name:25s}  → DROPPED")
    elif name in MERGE_MAP:
        dst = MERGE_MAP[name]
        print(f"    {c['id']:2d}  {name:25s}  → MERGED into '{dst}' (YOLO {name_to_yolo[dst]})")
    else:
        print(f"    {c['id']:2d}  {name:25s}  → YOLO {new_cat_remap[c['id']]}")

# Count annotations per new class to identify rare ones
ann_per_class_new = collections.Counter(
    new_cat_remap[a["category_id"]]
    for a in rubbish_coco["annotations"]
    if a["category_id"] in new_cat_remap
)
rare_classes = {cid for cid, cnt in ann_per_class_new.items() if cnt < RARE_THRESHOLD}
print(f"\n  Rare classes (<{RARE_THRESHOLD} ann): {[CLASS_NAMES[c] for c in sorted(rare_classes)]}")

class_copy_count = {
    cid: min(int(RARE_THRESHOLD / max(ann_per_class_new[cid], 1)) + 1, OVERSAMPLE_MAX)
    for cid in rare_classes
}

# =============================================================================
# SECTION 4 – COCO → YOLO SEGMENTATION LABELS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: CONVERT COCO → YOLO SEGMENTATION LABELS")
print("=" * 60)

out_dir = Path(YOLO_SEG_LABELS_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

def rle_to_polygon(seg_rle: dict, img_h: int, img_w: int) -> list | None:
    rle     = {"counts": seg_rle["counts"], "size": seg_rle["size"]}
    bm      = coco_mask.decode(rle).astype(np.uint8)
    contours, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    eps     = 0.005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, eps, True)
    if len(contour) < 3:
        return None
    pts = contour.squeeze()
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    return [round(min(max(float(v) / dim, 0), 1), 6)
            for x, y in pts
            for v, dim in ((x, img_w), (y, img_h))]

def poly_to_yolo(poly: list, img_w: int, img_h: int) -> list | None:
    norm = []
    for i in range(0, len(poly) - 1, 2):
        norm.extend([
            round(min(max(float(poly[i])   / img_w, 0), 1), 6),
            round(min(max(float(poly[i+1]) / img_h, 0), 1), 6),
        ])
    return norm if len(norm) >= 6 else None

img_info    = {i["id"]: i for i in rubbish_coco["images"]}
anns_by_img = collections.defaultdict(list)
for ann in rubbish_coco["annotations"]:
    if ann["category_id"] in new_cat_remap:
        anns_by_img[ann["image_id"]].append(ann)

ok = skip_bbox = fail = 0
for img_id, img in img_info.items():
    W, H  = img["width"], img["height"]
    lines = []
    for ann in anns_by_img.get(img_id, []):
        cid = new_cat_remap[ann["category_id"]]
        seg = ann.get("segmentation")
        if not seg:                             # no mask available → fall back to bbox
            x, y, w, h = [float(v) for v in ann["bbox"]]
            pts = [x/W, y/H, (x+w)/W, y/H, (x+w)/W, (y+h)/H, x/W, (y+h)/H]
            pts = [round(min(max(v, 0), 1), 6) for v in pts]
            lines.append(f"{cid} " + " ".join(map(str, pts)))
            skip_bbox += 1
        elif isinstance(seg, dict):             # RLE mask
            norm = rle_to_polygon(seg, H, W)
            if norm:
                lines.append(f"{cid} " + " ".join(map(str, norm)))
                ok += 1
            else:
                fail += 1
        elif isinstance(seg, list):             # Polygon mask
            best = max(seg, key=len) if seg else None
            norm = poly_to_yolo(best, W, H) if best else None
            if norm:
                lines.append(f"{cid} " + " ".join(map(str, norm)))
                ok += 1
            else:
                fail += 1
    fname = Path(img["file_name"]).stem + ".txt"
    (out_dir / fname).write_text("\n".join(lines))

print(f"  Polygon OK={ok:,}  fail={fail}  bbox_fallback={skip_bbox}")

# =============================================================================
# SECTION 5 – TRAIN/VAL SPLIT + OVERSAMPLE
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: TRAIN/VAL SPLIT + OVERSAMPLING")
print("=" * 60)

seg_split = Path(YOLO_SEG_SPLIT_DIR)
if seg_split.exists():
    shutil.rmtree(seg_split)

random.seed(SEED)
all_images = [
    img for img in (list(RUBBISH_DIR.glob("*.jpg")) + list(RUBBISH_DIR.glob("*.png")))
    if (out_dir / (img.stem + ".txt")).exists()
]
random.shuffle(all_images)
val_n      = int(len(all_images) * VAL_RATIO)
val_imgs   = all_images[:val_n]
train_imgs = all_images[val_n:]

for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
    (seg_split / split / "images").mkdir(parents=True, exist_ok=True)
    (seg_split / split / "labels").mkdir(parents=True, exist_ok=True)
    for img_path in imgs:
        shutil.copy(img_path, seg_split / split / "images" / img_path.name)
        lbl = out_dir / (img_path.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, seg_split / split / "labels" / lbl.name)

print(f"  Train: {len(train_imgs):,}  Val: {val_n:,}")

# Oversample rare classes
img_fname_map = {img["id"]: Path(img["file_name"]).stem for img in rubbish_coco["images"]}
img_to_rare: dict[str, set[int]] = collections.defaultdict(set)
for ann in rubbish_coco["annotations"]:
    if ann["category_id"] in new_cat_remap:
        yolo_cid = new_cat_remap[ann["category_id"]]
        if yolo_cid in rare_classes:
            img_to_rare[img_fname_map[ann["image_id"]]].add(yolo_cid)

train_img_dir = seg_split / "train" / "images"
train_lbl_dir = seg_split / "train" / "labels"
total_copied  = 0
for img_path in train_img_dir.iterdir():
    if img_path.stem not in img_to_rare:
        continue
    rare_in_img = img_to_rare[img_path.stem]
    n_copies    = max(class_copy_count.get(cid, 1) for cid in rare_in_img)
    lbl_path    = train_lbl_dir / (img_path.stem + ".txt")
    for i in range(1, n_copies + 1):
        stem = f"{img_path.stem}_os{i}"
        ni   = train_img_dir / f"{stem}{img_path.suffix}"
        nl   = train_lbl_dir / f"{stem}.txt"
        if not ni.exists():
            shutil.copy(img_path, ni)
            if lbl_path.exists():
                shutil.copy(lbl_path, nl)
            total_copied += 1

print(f"  Oversampled: +{total_copied} images")

# Add not-rubbish images as background (negative examples)
nr_images = list(NOTRUBBISH_DIR.glob("*.jpg")) + list(NOTRUBBISH_DIR.glob("*.png"))
random.seed(SEED); random.shuffle(nr_images)
nr_images = nr_images[:MAX_NOTRUBBISH]
nr_val_n  = int(len(nr_images) * VAL_RATIO)
for split, imgs in [("train", nr_images[nr_val_n:]), ("val", nr_images[:nr_val_n])]:
    for img_path in imgs:
        shutil.copy(img_path, seg_split / split / "images" / img_path.name)
        (seg_split / split / "labels" / (img_path.stem + ".txt")).touch()

print(f"  Not-rubbish background: {len(nr_images[nr_val_n:]):,} train  {nr_val_n} val")

# =============================================================================
# SECTION 6 – WRITE DATASET YAML
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: WRITE DATASET YAML")
print("=" * 60)

yaml_path = str(YOLO_SEG_SPLIT_DIR).replace("\\", "/")
yaml_content = f"""# Segmentation dataset – {len(CLASS_NAMES)} classes
# Drop: Clothes, Suitcase, Litter
# Merge: Wooden_trash→Wooden_crate, Plastic_bag→Garbage_bag
path: {yaml_path}
train: train/images
val:   val/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
with open(DATASET_SEG_YAML, "w", encoding="utf-8") as f:
    f.write(yaml_content.strip())

print(f"  Saved → {DATASET_SEG_YAML}")
print(f"\n  nc     : {len(CLASS_NAMES)}")
print(f"  names  : {CLASS_NAMES}")

# =============================================================================
# SECTION 7 – BUILD BINARY CLASSIFIER DATASET (RUBBISH / NOT-RUBBISH)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: BUILD BINARY CLASSIFIER DATASET (rubbish / notrubbish)")
print("=" * 60)

cls_split = Path(CLS_SPLIT_DIR)
if cls_split.exists():
    shutil.rmtree(cls_split)

random.seed(SEED)
for cls_name, src_dir in [("rubbish", RUBBISH_DIR), ("notrubbish", NOTRUBBISH_DIR)]:
    src  = Path(src_dir)
    imgs = list(src.glob("*.jpg")) + list(src.glob("*.png"))
    random.shuffle(imgs)
    v_n  = int(len(imgs) * VAL_RATIO)
    for part, part_imgs in [("val", imgs[:v_n]), ("train", imgs[v_n:])]:
        dst = cls_split / part / cls_name
        dst.mkdir(parents=True, exist_ok=True)
        for img in part_imgs:
            shutil.copy(img, dst / img.name)
    print(f"  {cls_name:12s}: train={len(imgs)-v_n:,}  val={v_n:,}")

print(f"\n  Dataset → {CLS_SPLIT_DIR}")

# =============================================================================
# SECTION 8 – TRAIN BINARY CLASSIFIER (YOLO11l-cls)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 8: TRAIN BINARY CLASSIFIER (YOLO11l-cls)")
print("=" * 60)
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM    : {vram:.1f} GB")

print(f"\n  Model   : {CLS_MODEL}")
print(f"  Imgsz   : {CLS_IMGSZ}")
print(f"  Batch   : {CLS_BATCH}")
print(f"  Epochs  : {CLS_EPOCHS}")
print(f"  Output  : {RUNS_DIR / CLS_RUN}")

# Clear stale cache files
for cache in Path(CLS_SPLIT_DIR).rglob("*.cache"):
    cache.unlink(missing_ok=True)

cls_model = YOLO(CLS_MODEL)

if __name__ == "__main__":
    cls_model.train(
        data      = str(CLS_SPLIT_DIR),
        epochs    = CLS_EPOCHS,
        imgsz     = CLS_IMGSZ,
        batch     = CLS_BATCH,
        device    = DEVICE,
        workers   = 0,          # Windows Python 3.14 requires workers=0
        project   = str(RUNS_DIR),
        name      = CLS_RUN,
        exist_ok  = True,
        optimizer = "AdamW",
        lr0       = 0.001,
        cos_lr    = True,
        patience  = 15,
        augment   = True,
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        fliplr    = 0.5,
        degrees   = 10,
        scale     = 0.5,
        erasing   = 0.3,        # random erasing to reduce overfitting
    )

    # ==========================================================================
    # SECTION 9 – VALIDATE CLASSIFIER
    # ==========================================================================

    print("\n" + "=" * 60)
    print("SECTION 9: VALIDATE CLASSIFIER")
    print("=" * 60)

    best_cls = RUNS_DIR / CLS_RUN / "weights" / "best.pt"
    if best_cls.exists():
        cls_val_model = YOLO(str(best_cls))
        cls_metrics   = cls_val_model.val(
            data    = str(CLS_SPLIT_DIR),
            imgsz   = CLS_IMGSZ,
            device  = DEVICE,
            workers = 0,
        )
        top1 = getattr(cls_metrics, "top1", None)
        top5 = getattr(cls_metrics, "top5", None)
        if top1 is not None:
            print(f"\n  Top-1 Accuracy: {top1:.4f}  ({top1*100:.2f}%)")
        if top5 is not None:
            print(f"  Top-5 Accuracy: {top5:.4f}  ({top5*100:.2f}%)")
    else:
        print(f"  Not found: {best_cls}")

    # ==========================================================================
    # SECTION 10 – TRAIN SEGMENTATION MODEL (YOLO11l-seg)
    # ==========================================================================

    print("\n" + "=" * 60)
    print("SECTION 10: TRAIN SEGMENTATION (YOLO11l-seg, 12 classes)")
    print("=" * 60)
    print(f"  Model   : {SEG_MODEL}")
    print(f"  Imgsz   : {SEG_IMGSZ}")
    print(f"  Batch   : {SEG_BATCH}")
    print(f"  Epochs  : {SEG_EPOCHS}")
    print(f"  Dataset : {DATASET_SEG_YAML}")
    print(f"  Output  : {RUNS_DIR / SEG_RUN}")

    seg_model = YOLO(SEG_MODEL)

    seg_model.train(
        data    = str(DATASET_SEG_YAML),
        epochs  = SEG_EPOCHS,
        imgsz   = SEG_IMGSZ,
        batch   = SEG_BATCH,
        workers = 0,
        device  = DEVICE,

        # ── Optimizer ─────────────────────────────────────────────────────────
        optimizer       = "AdamW",
        lr0             = 0.001,
        lrf             = 0.01,
        momentum        = 0.937,
        warmup_epochs   = 3.0,
        warmup_momentum = 0.8,
        cos_lr          = True,
        close_mosaic    = 10,

        # ── Augmentation ──────────────────────────────────────────────────────
        mosaic      = 1.0,
        mixup       = 0.1,
        copy_paste  = 0.3,
        flipud      = 0.0,
        fliplr      = 0.5,
        degrees     = 5.0,
        translate   = 0.1,
        scale       = 0.5,
        shear       = 1.0,
        perspective = 0.0,      # disabled – distorts polygon masks
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,

        # ── Loss weights ──────────────────────────────────────────────────────
        cls         = 0.5,
        box         = 7.5,

        # ── Regularisation ────────────────────────────────────────────────────
        weight_decay = 0.001,
        dropout      = 0.1,

        # ── Early stopping & checkpointing ────────────────────────────────────
        patience    = 20,
        save        = True,
        save_period = 10,
        exist_ok    = True,
        plots       = True,
        verbose     = True,

        # ── Output ────────────────────────────────────────────────────────────
        project = str(RUNS_DIR),
        name    = SEG_RUN,
    )

    # ==========================================================================
    # SECTION 11 – VALIDATE SEGMENTATION
    # ==========================================================================

    print("\n" + "=" * 60)
    print("SECTION 11: VALIDATE SEGMENTATION")
    print("=" * 60)

    best_seg = RUNS_DIR / SEG_RUN / "weights" / "best.pt"
    if not best_seg.exists():
        print(f"  Not found: {best_seg}")
    else:
        seg_val_model = YOLO(str(best_seg))
        metrics = seg_val_model.val(
            data    = str(DATASET_SEG_YAML),
            imgsz   = SEG_IMGSZ,
            conf    = 0.25,
            iou     = 0.5,
            device  = DEVICE,
            workers = 0,
            plots   = True,
        )

        print(f"\n  Box mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"  Box mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"  Seg mAP@0.5      : {metrics.seg.map50:.4f}")
        print(f"  Seg mAP@0.5:0.95 : {metrics.seg.map:.4f}")
        print(f"  Precision        : {metrics.box.mp:.4f}")
        print(f"  Recall           : {metrics.box.mr:.4f}")

        # Per-class Seg AP
        if hasattr(metrics.seg, "ap_class_index") and metrics.seg.ap_class_index is not None:
            print("\n  Per-class Seg AP@0.5 (high → low):")
            per_class = [
                (CLASS_NAMES[int(idx)], float(ap))
                for idx, ap in zip(metrics.seg.ap_class_index, metrics.seg.ap50)
            ]
            for cls, ap in sorted(per_class, key=lambda x: -x[1]):
                bar  = "█" * int(ap * 30)
                flag = "  ← needs improvement" if ap < 0.3 else ""
                print(f"    {cls:30s} {ap:.4f}  {bar}{flag}")

            # Per-class AP chart
            per_class_sorted = sorted(per_class, key=lambda x: -x[1])
            cls_labels = [c for c, _ in per_class_sorted]
            ap_values  = [a for _, a in per_class_sorted]
            bar_colors = ["#22c55e" if a >= 0.5 else "#f59e0b" if a >= 0.3 else "#ef4444"
                          for a in ap_values]

            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.bar(cls_labels, ap_values, color=bar_colors)
            ax2.axhline(0.5, color="green", linestyle="--", linewidth=1, label="0.5")
            ax2.axhline(0.3, color="orange", linestyle="--", linewidth=1, label="0.3")
            ax2.set_ylim(0, 1)
            ax2.set_xticks(range(len(cls_labels)))
            ax2.set_xticklabels(cls_labels, rotation=45, ha="right")
            ax2.set_title("Per-class Seg mAP@0.5 – YOLO11l-seg (12 cls)")
            ax2.set_ylabel("AP@0.5")
            ax2.legend()
            plt.tight_layout()
            plt.savefig(str(BASE_DIR / "per_class_ap.png"), dpi=150)
            plt.close()
            print("\n  Saved: per_class_ap.png")
