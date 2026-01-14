import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import yaml


@dataclass(frozen=True)
class Box:
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def _tile_starts(full: int, tile: int, overlap: float) -> list[int]:
    if full <= tile:
        return [0]
    overlap = max(0.0, min(0.95, float(overlap)))
    stride = max(1, int(round(tile * (1.0 - overlap))))
    n = int(math.ceil((full - tile) / stride)) + 1
    if n <= 1:
        return [0]
    last = full - tile
    starts = [int(round(i * last / (n - 1))) for i in range(n)]
    return sorted(set(starts))


def _read_yolo_labels(label_path: Path, *, image_w: int, image_h: int) -> list[Box]:
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for raw in label_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue

        x1 = (xc - w / 2.0) * float(image_w)
        y1 = (yc - h / 2.0) * float(image_h)
        x2 = (xc + w / 2.0) * float(image_w)
        y2 = (yc + h / 2.0) * float(image_h)

        x1 = max(0.0, min(float(image_w), x1))
        y1 = max(0.0, min(float(image_h), y1))
        x2 = max(0.0, min(float(image_w), x2))
        y2 = max(0.0, min(float(image_h), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(Box(cls_id=cls_id, x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


def _boxes_in_tile(
    boxes: Iterable[Box], *, x0: int, y0: int, tile_w: int, tile_h: int
) -> list[Box]:
    x1t = float(x0)
    y1t = float(y0)
    x2t = float(x0 + tile_w)
    y2t = float(y0 + tile_h)

    selected: list[Box] = []
    for box in boxes:
        if box.cx < x1t or box.cx >= x2t or box.cy < y1t or box.cy >= y2t:
            continue
        x1 = max(x1t, box.x1)
        y1 = max(y1t, box.y1)
        x2 = min(x2t, box.x2)
        y2 = min(y2t, box.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        selected.append(Box(cls_id=box.cls_id, x1=x1 - x1t, y1=y1 - y1t, x2=x2 - x1t, y2=y2 - y1t))
    return selected


def _write_yolo_labels(label_path: Path, boxes: Iterable[Box], *, image_w: int, image_h: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for box in boxes:
        x1 = max(0.0, min(float(image_w), box.x1))
        y1 = max(0.0, min(float(image_h), box.y1))
        x2 = max(0.0, min(float(image_w), box.x2))
        y2 = max(0.0, min(float(image_h), box.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        xc = (x1 + x2) / 2.0 / float(image_w)
        yc = (y1 + y2) / 2.0 / float(image_h)
        w = (x2 - x1) / float(image_w)
        h = (y2 - y1) / float(image_h)
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        lines.append(f"{box.cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _find_class_id(names: dict, query: str) -> Optional[int]:
    q = query.strip().lower()
    if not q or q in {"none", "null", "all"}:
        return None
    if q.isdigit():
        return int(q)

    for k, v in names.items():
        name = str(v).strip().lower()
        if name == q:
            return int(k)
    for k, v in names.items():
        name = str(v).strip().lower()
        if q in name:
            return int(k)
    return None


def _choose_best_tile(
    *, px: float, py: float, xs: list[int], ys: list[int], tile_w: int, tile_h: int
) -> tuple[int, int]:
    best: Optional[tuple[int, int]] = None
    best_d2 = float("inf")
    for y0 in ys:
        if py < y0 or py >= y0 + tile_h:
            continue
        cy = y0 + tile_h / 2.0
        for x0 in xs:
            if px < x0 or px >= x0 + tile_w:
                continue
            cx = x0 + tile_w / 2.0
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (x0, y0)
    if best is not None:
        return best

    x0 = min(xs, key=lambda v: abs(px - (v + tile_w / 2.0))) if xs else 0
    y0 = min(ys, key=lambda v: abs(py - (v + tile_h / 2.0))) if ys else 0
    return int(x0), int(y0)


def build_tiled_split(
    *,
    images_dir: Path,
    labels_dir: Path,
    out_root: Path,
    split: str,
    imgsz: int,
    tile_w: Optional[int],
    tile_h: Optional[int],
    overlap: float,
    focus_class_id: Optional[int],
    strategy: str,
    include_original: bool,
    limit: Optional[int],
) -> tuple[list[str], int]:
    split_images_out = out_root / "images" / split
    split_labels_out = out_root / "labels" / split
    split_images_out.mkdir(parents=True, exist_ok=True)
    split_labels_out.mkdir(parents=True, exist_ok=True)

    tile_w_eff = int(tile_w) if tile_w is not None else int(imgsz)
    tile_h_eff = int(tile_h) if tile_h is not None else int(round(imgsz * 0.75))

    image_paths = sorted(
        p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if limit is not None:
        image_paths = image_paths[: int(max(0, limit))]

    train_list: list[str] = []
    if include_original:
        train_list.extend(p.as_posix() for p in image_paths)

    tiles_written = 0
    for img_path in image_paths:
        img = cv2.imread(img_path.as_posix())
        if img is None:
            continue
        h, w = img.shape[:2]
        tw = int(min(max(1, tile_w_eff), w))
        th = int(min(max(1, tile_h_eff), h))
        xs = _tile_starts(w, tw, overlap)
        ys = _tile_starts(h, th, overlap)

        rel = img_path.relative_to(images_dir)
        label_path = labels_dir / rel.with_suffix(".txt")
        boxes = _read_yolo_labels(label_path, image_w=w, image_h=h)

        selected_tiles: list[tuple[int, int]]
        if strategy == "centered" and focus_class_id is not None:
            focus = [b for b in boxes if b.cls_id == int(focus_class_id)]
            chosen = {_choose_best_tile(px=b.cx, py=b.cy, xs=xs, ys=ys, tile_w=tw, tile_h=th) for b in focus}
            selected_tiles = sorted(chosen)
        else:
            selected_tiles = [(x0, y0) for y0 in ys for x0 in xs]

        for x0, y0 in selected_tiles:
            tile_boxes = _boxes_in_tile(boxes, x0=x0, y0=y0, tile_w=tw, tile_h=th)
            if focus_class_id is not None and not any(b.cls_id == int(focus_class_id) for b in tile_boxes):
                continue
            if not tile_boxes:
                continue

            tile_img = img[y0 : y0 + th, x0 : x0 + tw]
            out_img_path = (split_images_out / rel.parent / f"{rel.stem}_x{x0}_y{y0}{img_path.suffix}").resolve()
            out_lbl_path = (split_labels_out / rel.parent / f"{rel.stem}_x{x0}_y{y0}.txt").resolve()
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(out_img_path.as_posix(), tile_img)
            if not ok:
                continue
            _write_yolo_labels(out_lbl_path, tile_boxes, image_w=tw, image_h=th)
            train_list.append(out_img_path.as_posix())
            tiles_written += 1

    return train_list, tiles_written


def build_tiled_dataset(
    *,
    src_data_yaml: Path,
    out_root: Path,
    imgsz: int,
    tile_w: Optional[int],
    tile_h: Optional[int],
    overlap: float,
    focus_class: Optional[str],
    strategy: str,
    include_original: bool,
    splits: list[str],
    limit: Optional[int],
) -> Path:
    data = yaml.safe_load(src_data_yaml.read_text()) or {}
    src_base = Path(data.get("path", src_data_yaml.parent)).expanduser().resolve()
    names = data.get("names") or {}

    focus_class_id = None
    if focus_class is not None:
        if isinstance(names, dict):
            focus_class_id = _find_class_id(names, focus_class)
        else:
            focus_class_id = None

    out_root = out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    out_data = dict(data)
    out_data["path"] = out_root.as_posix()

    processed: set[str] = set()
    for split in splits:
        key = split
        if key not in data:
            continue
        processed.add(key)
        split_images = Path(data[key])
        if not split_images.is_absolute():
            split_images = (src_base / split_images).resolve()

        try:
            rel_from_images = split_images.relative_to(src_base / "images")
            split_labels = (src_base / "labels" / rel_from_images).resolve()
        except ValueError:
            split_labels = (src_base / "labels" / split).resolve()

        train_list, tiles_written = build_tiled_split(
            images_dir=split_images,
            labels_dir=split_labels,
            out_root=out_root,
            split=split,
            imgsz=imgsz,
            tile_w=tile_w,
            tile_h=tile_h,
            overlap=overlap,
            focus_class_id=focus_class_id,
            strategy=strategy,
            include_original=include_original,
            limit=limit,
        )

        list_path = out_root / f"{split}.txt"
        list_path.write_text("\n".join(train_list) + ("\n" if train_list else ""))
        out_data[key] = list_path.as_posix()
        print(f"[{split}] tiles_written={tiles_written} total_images={len(train_list)} list={list_path}")

    for key in ("train", "val", "test"):
        if key in processed or key not in data:
            continue
        value = data.get(key)
        if not isinstance(value, str):
            continue
        split_images = Path(value)
        if not split_images.is_absolute():
            split_images = (src_base / split_images).resolve()
        out_data[key] = split_images.as_posix()

    out_yaml = out_root / "data.yaml"
    out_yaml.write_text(yaml.safe_dump(out_data, sort_keys=False))
    return out_yaml


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_src = repo_root / "tracking_dataset" / "data.yaml"
    default_out = repo_root / "tracking_dataset_tiled"

    parser = argparse.ArgumentParser(
        description="Generate a tiled YOLO dataset (adds tile crops; can optionally keep original images)."
    )
    parser.add_argument("--src-data", type=Path, default=default_src, help="Path to source data.yaml.")
    parser.add_argument("--out", type=Path, default=default_out, help="Output dataset directory.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training imgsz (used for tile defaults).")
    parser.add_argument("--tile-w", type=int, default=None, help="Tile width in source pixels (default: imgsz).")
    parser.add_argument(
        "--tile-h",
        type=int,
        default=None,
        help="Tile height in source pixels (default: 0.75 * imgsz).",
    )
    parser.add_argument("--overlap", type=float, default=0.2, help="Tile overlap fraction in [0, 0.95).")
    parser.add_argument(
        "--focus-class",
        default="ball",
        help="Only keep tiles containing this class (name or id). Use 'none' to disable.",
    )
    parser.add_argument(
        "--strategy",
        choices=("centered", "grid"),
        default="centered",
        help="How to select tiles when focus-class is set.",
    )
    parser.set_defaults(include_original=True)
    parser.add_argument(
        "--no-include-original",
        dest="include_original",
        action="store_false",
        help="Only use tiles (do not include original images in split lists).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Splits to process (must exist as keys in data.yaml, e.g. train val).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images per split (debug).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    focus = None if str(args.focus_class).strip().lower() in {"none", "null", "all"} else str(args.focus_class)
    out_yaml = build_tiled_dataset(
        src_data_yaml=Path(args.src_data).expanduser().resolve(),
        out_root=Path(args.out),
        imgsz=int(args.imgsz),
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        overlap=float(args.overlap),
        focus_class=focus,
        strategy=str(args.strategy),
        include_original=bool(args.include_original),
        splits=[str(s) for s in args.splits],
        limit=args.limit,
    )
    print(f"Wrote: {out_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
