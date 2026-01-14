import argparse
import json
import os
import shutil
from pathlib import Path

TARGET_CLASSES = ["player", "goalkeeper", "referee", "ball"]


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def yolo_bbox_line(class_idx, bbox, img_w, img_h):
    if bbox is None:
        return None
    if img_w <= 0 or img_h <= 0:
        return None
    if "x_center" in bbox and "y_center" in bbox:
        x_center = float(bbox["x_center"])
        y_center = float(bbox["y_center"])
    else:
        x_center = float(bbox["x"]) + float(bbox["w"]) / 2.0
        y_center = float(bbox["y"]) + float(bbox["h"]) / 2.0
    w = float(bbox["w"])
    h = float(bbox["h"])
    x_center = clamp(x_center / float(img_w))
    y_center = clamp(y_center / float(img_h))
    w = clamp(w / float(img_w))
    h = clamp(h / float(img_h))
    return f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


def safe_copy(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_labels(label_path: Path, class_name_to_index):
    data = json.loads(label_path.read_text())
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    category_id_to_name = {}
    for category in categories:
        try:
            category_id = int(category["id"])
        except (KeyError, TypeError, ValueError):
            continue
        category_id_to_name[category_id] = category.get("name")

    image_id_to_info = {}
    for image in images:
        image_id = str(image.get("image_id"))
        if image_id:
            image_id_to_info[image_id] = image

    labels_by_file = {}
    for ann in annotations:
        try:
            category_id = int(ann.get("category_id"))
        except (TypeError, ValueError):
            continue
        category_name = category_id_to_name.get(category_id)
        if category_name not in class_name_to_index:
            continue
        image_id = str(ann.get("image_id"))
        image_info = image_id_to_info.get(image_id)
        if not image_info:
            continue
        bbox = ann.get("bbox_image")
        line = yolo_bbox_line(
            class_name_to_index[category_name],
            bbox,
            image_info.get("width", 0),
            image_info.get("height", 0),
        )
        if line is None:
            continue
        file_name = image_info.get("file_name")
        if not file_name:
            continue
        labels_by_file.setdefault(file_name, []).append(line)

    return images, labels_by_file


def write_data_yaml(out_root: Path, class_names):
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(class_names)],
                "",
            ]
        )
    )


def process_split(split_dir: Path, split_name: str, out_root: Path, mode: str, class_name_to_index):
    if not split_dir.exists():
        print(f"Skipping missing split: {split_dir}")
        return 0, 0

    images_written = 0
    labels_written = 0

    for sequence_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label_path = sequence_dir / "Labels-GameState.json"
        img_dir = sequence_dir / "img1"
        if not label_path.exists() or not img_dir.exists():
            continue

        images, labels_by_file = load_labels(label_path, class_name_to_index)

        out_img_dir = out_root / "images" / split_name / sequence_dir.name
        out_lbl_dir = out_root / "labels" / split_name / sequence_dir.name
        ensure_dir(out_img_dir)
        ensure_dir(out_lbl_dir)

        for image in images:
            if not image.get("is_labeled", True):
                continue
            file_name = image.get("file_name")
            if not file_name:
                continue
            src_img = img_dir / file_name
            if not src_img.exists():
                continue
            dst_img = out_img_dir / file_name
            if mode == "copy":
                safe_copy(src_img, dst_img)
            else:
                safe_symlink(src_img.resolve(), dst_img)
            images_written += 1

            label_lines = labels_by_file.get(file_name, [])
            label_path = out_lbl_dir / f"{Path(file_name).stem}.txt"
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
            labels_written += 1

    return images_written, labels_written


def main():
    parser = argparse.ArgumentParser(
        description="Build a YOLO-format tracking dataset from SN-GSR-2025."
    )
    parser.add_argument("--src", default="SN-GSR-2025", help="Source dataset root.")
    parser.add_argument(
        "--out",
        default="SN-GSR-2025-tracking",
        help="Output dataset root for YOLO tracking data.",
    )
    parser.add_argument(
        "--splits",
        default="train,valid",
        help="Comma-separated splits to include (e.g. train,valid).",
    )
    parser.add_argument(
        "--mode",
        choices=["link", "copy"],
        default="link",
        help="Use symlinks or copy images into the output dataset.",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    ensure_dir(out_root)

    class_name_to_index = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
    split_map = {"train": "train", "valid": "val"}

    total_images = 0
    total_labels = 0

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split_dir = src_root / split
        out_split_name = split_map.get(split, split)
        images_written, labels_written = process_split(
            split_dir, out_split_name, out_root, args.mode, class_name_to_index
        )
        total_images += images_written
        total_labels += labels_written
        print(
            f"{split}: wrote {images_written} images and {labels_written} label files."
        )

    write_data_yaml(out_root, TARGET_CLASSES)
    print(f"Done. Total images: {total_images}, labels: {total_labels}")
    print(f"YOLO data.yaml: {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
