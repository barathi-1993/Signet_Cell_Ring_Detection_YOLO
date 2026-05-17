"""
python tools/convert_digestpath_to_yolo.py
"""
from pathlib import Path
import random
import shutil
import math
import yaml
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict

# =========================
# Config
# =========================
DATASET_ROOT = Path("/data_64T_1/Barathi/Projects/SRC_detection/digestpath_dataset")
POS_DIR = DATASET_ROOT / "sig-train-pos"
NEG_DIR = DATASET_ROOT / "sig-train-neg"

OUT_ROOT = Path("/data_64T_1/Barathi/Projects/SRC_detection/digestpath_dataset/digestpath_yolo")
IMG_OUT = OUT_ROOT / "images"
LBL_OUT = OUT_ROOT / "labels"

SPLITS = ["train", "val"]

RANDOM_SEED = 42
VAL_RATIO = 0.2

TILE_SIZE = 1024
OVERLAP = 256   # 25% overlap
MIN_BOX_AREA = 16  # discard tiny fragments after clipping
MIN_VISIBLE_FRAC = 0.3  # keep box in tile only if >=30% of original area remains

# negative tile control
KEEP_ALL_NEG_TILES = False
NEG_TO_POS_TILE_RATIO = 2.0  # if KEEP_ALL_NEG_TILES=False, sample negatives up to this ratio

CLASS_ID = 0
CLASS_NAME = "SRC"

# =========================
# XML parser
# =========================
def parse_boxes(xml_path: Path):
    boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall(".//object"):
        bnd = obj.find(".//bndbox")
        if bnd is not None:
            xmin = int(float(bnd.findtext("xmin", "0")))
            ymin = int(float(bnd.findtext("ymin", "0")))
            xmax = int(float(bnd.findtext("xmax", "0")))
            ymax = int(float(bnd.findtext("ymax", "0")))
            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))

    if not boxes:
        for bnd in root.findall(".//bndbox"):
            xmin = int(float(bnd.findtext("xmin", "0")))
            ymin = int(float(bnd.findtext("ymin", "0")))
            xmax = int(float(bnd.findtext("xmax", "0")))
            ymax = int(float(bnd.findtext("ymax", "0")))
            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))

    return boxes

# =========================
# Geometry helpers
# =========================
def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def clip_box_to_tile(box, tile_box):
    bx1, by1, bx2, by2 = box
    tx1, ty1, tx2, ty2 = tile_box

    cx1 = max(bx1, tx1)
    cy1 = max(by1, ty1)
    cx2 = min(bx2, tx2)
    cy2 = min(by2, ty2)

    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return (cx1, cy1, cx2, cy2)

def shift_box_to_tile_coords(box, tile_x, tile_y):
    x1, y1, x2, y2 = box
    return (x1 - tile_x, y1 - tile_y, x2 - tile_x, y2 - tile_y)

def xyxy_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h

def make_tile_positions(full_size, tile_size, overlap):
    stride = tile_size - overlap
    assert stride > 0, "overlap must be smaller than tile_size"

    if full_size <= tile_size:
        return [0]

    positions = list(range(0, full_size - tile_size + 1, stride))
    if positions[-1] != full_size - tile_size:
        positions.append(full_size - tile_size)
    return positions

# =========================
# Splitting
# =========================
def split_files(pos_imgs, neg_imgs, val_ratio=0.2, seed=42):
    rng = random.Random(seed)

    pos_imgs = pos_imgs[:]
    neg_imgs = neg_imgs[:]

    rng.shuffle(pos_imgs)
    rng.shuffle(neg_imgs)

    n_pos_val = max(1, round(len(pos_imgs) * val_ratio))
    n_neg_val = max(1, round(len(neg_imgs) * val_ratio))

    pos_val = pos_imgs[:n_pos_val]
    pos_train = pos_imgs[n_pos_val:]

    neg_val = neg_imgs[:n_neg_val]
    neg_train = neg_imgs[n_neg_val:]

    return {
        "train": {"pos": pos_train, "neg": neg_train},
        "val": {"pos": pos_val, "neg": neg_val},
    }

# =========================
# Tiling and saving
# =========================
def save_tile_and_label(tile_img, label_lines, out_img_path, out_lbl_path):
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

    tile_img.save(out_img_path, quality=95)
    with open(out_lbl_path, "w", encoding="utf-8") as f:
        for line in label_lines:
            f.write(line + "\n")

def process_positive_image(img_path, split_name, stats):
    xml_path = img_path.with_suffix(".xml")
    boxes = parse_boxes(xml_path)

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    xs = make_tile_positions(W, TILE_SIZE, OVERLAP)
    ys = make_tile_positions(H, TILE_SIZE, OVERLAP)

    pos_tiles = []
    neg_tiles = []

    for y in ys:
        for x in xs:
            tile_box = (x, y, x + TILE_SIZE, y + TILE_SIZE)
            label_lines = []

            for box in boxes:
                clipped = clip_box_to_tile(box, tile_box)
                if clipped is None:
                    continue

                original_area = box_area(box)
                clipped_area = box_area(clipped)

                if clipped_area < MIN_BOX_AREA:
                    continue
                if original_area == 0:
                    continue
                if clipped_area / original_area < MIN_VISIBLE_FRAC:
                    continue

                shifted = shift_box_to_tile_coords(clipped, x, y)
                xc, yc, w, h = xyxy_to_yolo(shifted, TILE_SIZE, TILE_SIZE)

                # guard against numerical issues
                if w <= 0 or h <= 0:
                    continue
                if not (0 <= xc <= 1 and 0 <= yc <= 1):
                    continue

                label_lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
            tile_name = f"{img_path.stem}__x{x}_y{y}.jpg"
            out_img = IMG_OUT / split_name / tile_name
            out_lbl = LBL_OUT / split_name / f"{Path(tile_name).stem}.txt"

            if len(label_lines) > 0:
                pos_tiles.append((tile, label_lines, out_img, out_lbl))
            else:
                neg_tiles.append((tile, label_lines, out_img, out_lbl))

    # save all positive tiles
    for tile, label_lines, out_img, out_lbl in pos_tiles:
        save_tile_and_label(tile, label_lines, out_img, out_lbl)

    # controlled saving of negative-only tiles from positive-source images
    if KEEP_ALL_NEG_TILES:
        selected_neg_tiles = neg_tiles
    else:
        max_neg = int(len(pos_tiles) * NEG_TO_POS_TILE_RATIO)
        if len(pos_tiles) == 0:
            max_neg = min(len(neg_tiles), 4)
        random.shuffle(neg_tiles)
        selected_neg_tiles = neg_tiles[:max_neg]

    for tile, label_lines, out_img, out_lbl in selected_neg_tiles:
        save_tile_and_label(tile, label_lines, out_img, out_lbl)

    stats[split_name]["pos_parent_images"] += 1
    stats[split_name]["pos_tiles"] += len(pos_tiles)
    stats[split_name]["neg_tiles_from_pos"] += len(selected_neg_tiles)
    stats[split_name]["boxes"] += sum(len(lbls) for _, lbls, _, _ in pos_tiles)

def process_negative_image(img_path, split_name, stats):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    xs = make_tile_positions(W, TILE_SIZE, OVERLAP)
    ys = make_tile_positions(H, TILE_SIZE, OVERLAP)

    neg_tiles = []
    for y in ys:
        for x in xs:
            tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
            tile_name = f"{img_path.stem}__x{x}_y{y}.jpg"
            out_img = IMG_OUT / split_name / tile_name
            out_lbl = LBL_OUT / split_name / f"{Path(tile_name).stem}.txt"
            neg_tiles.append((tile, [], out_img, out_lbl))

    # sample negative tiles if requested
    if KEEP_ALL_NEG_TILES:
        selected = neg_tiles
    else:
        # cap negatives based on current positive count in this split
        current_pos_tiles = stats[split_name]["pos_tiles"]
        target_total_neg = int(max(current_pos_tiles, 1) * NEG_TO_POS_TILE_RATIO)
        already_neg = stats[split_name]["neg_tiles_from_pos"] + stats[split_name]["neg_tiles_from_neg"]
        remaining_budget = max(0, target_total_neg - already_neg)

        random.shuffle(neg_tiles)
        selected = neg_tiles[:remaining_budget]

    for tile, label_lines, out_img, out_lbl in selected:
        save_tile_and_label(tile, label_lines, out_img, out_lbl)

    stats[split_name]["neg_parent_images"] += 1
    stats[split_name]["neg_tiles_from_neg"] += len(selected)

# =========================
# YAML writer
# =========================
def write_yaml():
    yaml_path = OUT_ROOT / "digestpath.yaml"
    data = {
        "path": str(OUT_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": [CLASS_NAME],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path

# =========================
# Main
# =========================
def clean_output_dirs():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in SPLITS:
        (IMG_OUT / split).mkdir(parents=True, exist_ok=True)
        (LBL_OUT / split).mkdir(parents=True, exist_ok=True)

def main():
    random.seed(RANDOM_SEED)
    clean_output_dirs()

    pos_imgs = sorted(POS_DIR.glob("*.jpeg"))
    neg_imgs = sorted(NEG_DIR.glob("*.jpeg"))

    split_map = split_files(pos_imgs, neg_imgs, val_ratio=VAL_RATIO, seed=RANDOM_SEED)

    stats = defaultdict(lambda: defaultdict(int))

    for split_name in SPLITS:
        # process positives first so negative sampling budget can follow positive count
        for img_path in split_map[split_name]["pos"]:
            process_positive_image(img_path, split_name, stats)

        for img_path in split_map[split_name]["neg"]:
            process_negative_image(img_path, split_name, stats)

    yaml_path = write_yaml()

    print("=== Conversion complete ===")
    for split_name in SPLITS:
        print(f"\n[{split_name}]")
        print(f"Positive parent images     : {stats[split_name]['pos_parent_images']}")
        print(f"Negative parent images     : {stats[split_name]['neg_parent_images']}")
        print(f"Positive tiles             : {stats[split_name]['pos_tiles']}")
        print(f"Negative tiles from pos    : {stats[split_name]['neg_tiles_from_pos']}")
        print(f"Negative tiles from neg    : {stats[split_name]['neg_tiles_from_neg']}")
        print(f"Boxes kept                 : {stats[split_name]['boxes']}")

    print(f"\nYOLO dataset root: {OUT_ROOT.resolve()}")
    print(f"YAML file        : {yaml_path.resolve()}")

if __name__ == "__main__":
    main()