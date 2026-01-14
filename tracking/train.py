import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "tracking_dataset" / "data.yaml"

# Tuned for an L4 GPU: larger model + higher resolution for small objects.
MODEL_WEIGHTS = "yolo11s.pt"
EPOCHS = 150
IMG_SIZE = 960
BATCH = 32
RUN_NAME = "l4_y11s_960"


def _format_table(rows: Iterable[dict[str, Any]], headers: list[str]) -> str:
    rows = list(rows)
    if not rows:
        return ""

    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    def fmt_cell(h: str, v: Any) -> str:
        s = str(v if v is not None else "")
        if h == "Class":
            return s.ljust(widths[h])
        return s.rjust(widths[h])

    lines = []
    lines.append("  ".join(fmt_cell(h, h) for h in headers))
    lines.append("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        lines.append("  ".join(fmt_cell(h, row.get(h, "")) for h in headers))
    return "\n".join(lines)


def _write_per_class_metrics(save_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)

    json_path = save_dir / "per_class_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")

    csv_path = save_dir / "per_class_metrics.csv"
    headers = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def _per_class_summary(metrics: Any, *, decimals: int) -> Optional[list[dict[str, Any]]]:
    if metrics is None:
        return None
    summary = getattr(metrics, "summary", None)
    if not callable(summary):
        return None
    try:
        rows = summary(decimals=int(decimals))
    except TypeError:
        rows = summary()
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows
    return None


def _default_tile_out(
    *,
    imgsz: int,
    tile_w: int,
    tile_h: int,
    overlap: float,
    focus_class: str,
    strategy: str,
    include_original: bool,
) -> Path:
    ov = f"{overlap:g}".replace(".", "p")
    mix = "mix" if include_original else "tiles"
    safe_focus = "".join(c for c in focus_class.lower() if c.isalnum() or c in {"-", "_"}).strip("_-")
    safe_focus = safe_focus or "all"
    return REPO_ROOT / (
        f"tracking_dataset_tiled_{safe_focus}_{mix}_img{imgsz}_tw{tile_w}_th{tile_h}_ov{ov}_{strategy}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model (with optional tiled crops).")
    parser.add_argument("--weights", default=MODEL_WEIGHTS, help="Base weights or checkpoint (.pt).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="Source data.yaml.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Training image size.")
    parser.add_argument("--batch", type=int, default=BATCH, help="Batch size.")
    parser.add_argument("--device", default="0", help="Device string passed to Ultralytics (e.g. 0, cpu).")
    parser.add_argument("--name", default=RUN_NAME, help="Run name under runs/train.")

    parser.set_defaults(tiled=True)
    parser.add_argument(
        "--no-tiled",
        dest="tiled",
        action="store_false",
        help="Disable tiled training augmentation (use original dataset only).",
    )
    parser.add_argument("--tile-out", type=Path, default=None, help="Output folder for tiled dataset cache.")
    parser.add_argument("--tile-w", type=int, default=None, help="Tile width in source pixels (default: imgsz).")
    parser.add_argument(
        "--tile-h",
        type=int,
        default=None,
        help="Tile height in source pixels (default: 0.75 * imgsz).",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.2, help="Tile overlap fraction in [0, 0.95).")
    parser.add_argument(
        "--tile-focus-class",
        default="ball",
        help="Only add tiles that contain this class (name or id). Use 'none' to disable.",
    )
    parser.add_argument(
        "--tile-strategy",
        choices=("centered", "grid"),
        default="centered",
        help="Tile selection strategy when --tile-focus-class is set.",
    )
    parser.add_argument(
        "--tile-splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to tile (must exist as keys in data.yaml).",
    )
    parser.add_argument(
        "--tile-only",
        action="store_true",
        help="Train only on tiles (do not include original full images in the train list).",
    )
    parser.add_argument("--tile-rebuild", action="store_true", help="Force rebuilding the tiled dataset cache.")
    parser.add_argument("--tile-limit", type=int, default=None, help="Limit images when building tiles (debug).")

    parser.set_defaults(per_class_eval=True)
    parser.add_argument(
        "--no-per-class-eval",
        dest="per_class_eval",
        action="store_false",
        help="Disable per-class metrics summary after training.",
    )
    parser.add_argument(
        "--per-class-eval-decimals",
        type=int,
        default=5,
        help="Decimal places for per-class metrics.",
    )
    parser.add_argument(
        "--per-class-eval-out",
        type=Path,
        default=None,
        help="Directory to write per_class_metrics.{json,csv} (default: training run directory).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    run_name = str(args.name)

    if args.tiled:
        try:
            from tile_dataset import build_tiled_dataset
        except ImportError:  # pragma: no cover
            from tracking.tile_dataset import build_tiled_dataset

        imgsz = int(args.imgsz)
        tile_w = int(args.tile_w) if args.tile_w is not None else imgsz
        tile_h = int(args.tile_h) if args.tile_h is not None else int(round(imgsz * 0.75))
        overlap = float(args.tile_overlap)
        include_original = not bool(args.tile_only)
        focus_class = str(args.tile_focus_class)
        focus_class = None if focus_class.strip().lower() in {"none", "null", "all"} else focus_class
        strategy = str(args.tile_strategy)

        out_root = (
            Path(args.tile_out).expanduser().resolve()
            if args.tile_out is not None
            else _default_tile_out(
                imgsz=imgsz,
                tile_w=tile_w,
                tile_h=tile_h,
                overlap=overlap,
                focus_class=str(focus_class or "all"),
                strategy=strategy,
                include_original=include_original,
            )
        )
        out_yaml = out_root / "data.yaml"
        if args.tile_rebuild or not out_yaml.exists():
            out_yaml = build_tiled_dataset(
                src_data_yaml=data_yaml,
                out_root=out_root,
                imgsz=imgsz,
                tile_w=tile_w,
                tile_h=tile_h,
                overlap=overlap,
                focus_class=focus_class,
                strategy=strategy,
                include_original=include_original,
                splits=[str(s) for s in args.tile_splits],
                limit=args.tile_limit,
            )
        data_yaml = out_yaml
        if run_name == RUN_NAME:
            run_name = f"{RUN_NAME}_tiled"

    model = YOLO(str(args.weights))
    metrics = model.train(
        data=data_yaml.as_posix(),
        epochs=int(args.epochs),
        batch=int(args.batch),
        imgsz=int(args.imgsz),
        device=str(args.device),
        save=True,
        save_period=1,
        name=run_name,
        pretrained=True,
        val=True,
        cos_lr=True,
        mixup=0.1,
        copy_paste=0.1,
    )

    if args.per_class_eval:
        rows = _per_class_summary(metrics, decimals=int(args.per_class_eval_decimals))
        val_error: Optional[Exception] = None
        if not rows:
            try:
                metrics = model.val(
                    data=data_yaml.as_posix(),
                    imgsz=int(args.imgsz),
                    batch=int(args.batch),
                    device=str(args.device),
                )
                rows = _per_class_summary(metrics, decimals=int(args.per_class_eval_decimals))
            except Exception as exc:  # pragma: no cover
                val_error = exc
        if rows:
            headers = ["Class", "Images", "Instances", "Box-P", "Box-R", "Box-F1", "mAP50", "mAP50-95"]
            print("\nPer-class evaluation (val):")
            print(_format_table(rows, headers=headers))

            if args.per_class_eval_out is not None:
                out_dir = Path(args.per_class_eval_out).expanduser().resolve()
            else:
                trainer = getattr(model, "trainer", None)
                out_dir = Path(getattr(trainer, "save_dir", REPO_ROOT / "runs")).resolve()

            json_path, csv_path = _write_per_class_metrics(out_dir, rows)
            print(f"Wrote per-class metrics: {json_path}")
            print(f"Wrote per-class metrics: {csv_path}")
        elif val_error is not None:
            print(f"\nPer-class evaluation skipped (validation failed: {val_error})")
        else:
            print("\nPer-class evaluation skipped (no metrics returned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
