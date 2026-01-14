import argparse
import math
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PALETTE = [
    (0, 165, 255),  # orange
    (0, 209, 178),  # teal
    (255, 128, 0),  # blue
    (80, 200, 120),  # green
    (255, 191, 0),  # cyan
    (180, 105, 255),  # pink
]
BALL_COLOR = (0, 255, 255)  # yellow


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained YOLO model on a video file or a frames directory."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained weights (.pt).",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a video file, frames directory, or image glob.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device to run on (e.g. cpu, cuda:0, mps).",
    )
    parser.add_argument(
        "--project",
        default="runs/predict",
        help="Project directory for outputs.",
    )
    parser.add_argument(
        "--name",
        default="exp",
        help="Run name for outputs.",
    )
    parser.set_defaults(save=True)
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Disable saving annotated outputs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show results in a window during inference.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detections to .txt files.",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Save confidence scores in .txt labels.",
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped detections.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=25.0,
        help="FPS for output video when the source has no FPS metadata.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Also save annotated frame images alongside the video.",
    )
    parser.set_defaults(tiled=True)
    parser.add_argument(
        "--no-tiled",
        dest="tiled",
        action="store_false",
        help="Disable tiled (sliced) inference.",
    )
    parser.add_argument(
        "--tile-w",
        type=int,
        default=None,
        help="Tile width in source pixels (default: imgsz).",
    )
    parser.add_argument(
        "--tile-h",
        type=int,
        default=None,
        help="Tile height in source pixels (default: 0.75 * imgsz).",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.2,
        help="Tile overlap fraction in [0, 0.95).",
    )
    parser.add_argument(
        "--tile-batch",
        type=int,
        default=8,
        help="Tiles per model forward pass.",
    )
    return parser.parse_args()


def _class_name(names, cls_id):
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    if isinstance(names, (list, tuple)):
        if 0 <= cls_id < len(names):
            return names[cls_id]
    return str(cls_id)


def _is_ball_class(name):
    name = name.lower().replace(" ", "")
    return "ball" in name


def _color_for_class(cls_id):
    if cls_id is None or cls_id < 0:
        return PALETTE[0]
    return PALETTE[cls_id % len(PALETTE)]


def _line_thickness(shape):
    height, width = shape[:2]
    return max(2, int(round(min(height, width) / 320)))


def _draw_detections(image, result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return image

    xyxy = boxes.xyxy
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu()
    if hasattr(xyxy, "numpy"):
        xyxy = xyxy.numpy()
    else:
        xyxy = np.asarray(xyxy)

    cls = boxes.cls
    if hasattr(cls, "cpu"):
        cls = cls.cpu()
    if hasattr(cls, "numpy"):
        cls = cls.numpy()
    else:
        cls = np.asarray(cls)
    cls = cls.astype(int)

    height, width = image.shape[:2]
    thickness = _line_thickness(image.shape)
    outline = thickness + 2
    line_type = cv2.LINE_AA

    for idx, coords in enumerate(xyxy):
        if coords is None or len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cls_id = int(cls[idx]) if idx < len(cls) else -1
        name = _class_name(getattr(result, "names", None), cls_id)
        if _is_ball_class(name):
            color = BALL_COLOR
            cx = int(round((x1 + x2) / 2))
            cy = int(round((y1 + y2) / 2))
            radius = max(2, int(round(min(x2 - x1, y2 - y1) / 2)))
            cv2.circle(image, (cx, cy), radius, (0, 0, 0), outline, line_type)
            cv2.circle(image, (cx, cy), radius, color, thickness, line_type)
        else:
            color = _color_for_class(cls_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), outline, line_type)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, line_type)

    return image


def _resolve_save_dir(model, args, result=None):
    save_dir = getattr(result, "save_dir", None) if result is not None else None
    if save_dir is None and getattr(model, "predictor", None):
        save_dir = getattr(model.predictor, "save_dir", None)
    if save_dir is None:
        save_dir = Path(args.project) / args.name
    return Path(save_dir)


def _result_stem(result, model, frame_index):
    path = Path(getattr(result, "path", "frame"))
    stem = path.stem or "frame"
    predictor = getattr(model, "predictor", None)
    dataset = getattr(predictor, "dataset", None) if predictor else None
    mode = getattr(dataset, "mode", "image") if dataset else "image"
    if mode != "image":
        stem = f"{stem}_{frame_index + 1}"
    return stem


def _resolve_fps(model, args):
    fps = None
    predictor = getattr(model, "predictor", None)
    dataset = getattr(predictor, "dataset", None) if predictor else None
    if dataset is not None:
        fps = getattr(dataset, "fps", None)
        if isinstance(fps, (list, tuple)):
            fps = fps[0] if fps else None
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = None
    if not fps or fps <= 0:
        fps = args.video_fps
    return fps


def _init_video_writer(output_path, fps, frame_shape):
    height, width = frame_shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    return writer


class _BoxesView:
    def __init__(
        self, xyxy: np.ndarray, cls: np.ndarray, conf: Optional[np.ndarray] = None
    ):
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf

    def __len__(self) -> int:
        return 0 if self.xyxy is None else int(self.xyxy.shape[0])


class _FrameResult:
    def __init__(self, *, orig_img: np.ndarray, path: str, names, boxes: _BoxesView):
        self.orig_img = orig_img
        self.path = path
        self.names = names
        self.boxes = boxes


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def _iter_images_from_dir(directory: Path) -> Iterator[Tuple[np.ndarray, str]]:
    paths = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in directory: {directory}")
    for path in paths:
        img = cv2.imread(path.as_posix())
        if img is None:
            continue
        yield img, path.as_posix()


def _iter_images_from_glob(pattern: str) -> Iterator[Tuple[np.ndarray, str]]:
    import glob

    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    for path in paths:
        if path.suffix.lower() not in _IMAGE_EXTS:
            continue
        img = cv2.imread(path.as_posix())
        if img is None:
            continue
        yield img, path.as_posix()


def _open_video(source: Path) -> Tuple[cv2.VideoCapture, float]:
    cap = cv2.VideoCapture(source.as_posix())
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = 0.0
    if not fps or fps <= 0:
        fps = 0.0
    return cap, fps


def _iter_source_frames(source: str) -> Tuple[Iterator[Tuple[np.ndarray, str]], bool, float]:
    """
    Returns (frames_iter, is_video, fps).

    frames_iter yields (frame_bgr, path_str) where path_str is either the image path
    or the video path.
    """
    path = Path(source)
    if path.exists():
        if path.is_dir():
            return _iter_images_from_dir(path), False, 0.0
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTS:
            cap, fps = _open_video(path)

            def _iter() -> Iterator[Tuple[np.ndarray, str]]:
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        yield frame, path.as_posix()
                finally:
                    cap.release()

            return _iter(), True, fps
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
            img = cv2.imread(path.as_posix())
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {path}")
            return iter([(img, path.as_posix())]), False, 0.0

    return _iter_images_from_glob(source), False, 0.0


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


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]

        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= float(iou_thr)]

    return torch.tensor(keep, dtype=torch.long)


def _nms_by_class(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float
) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    keep_all: list[int] = []
    for cls_id in np.unique(classes):
        idxs = np.flatnonzero(classes == cls_id)
        kept_local = _nms(
            torch.from_numpy(boxes[idxs]).float(),
            torch.from_numpy(scores[idxs]).float(),
            iou_thr=float(iou_thr),
        ).numpy()
        keep_all.extend(idxs[kept_local].tolist())

    keep_all.sort(key=lambda i: float(scores[i]), reverse=True)
    return np.asarray(keep_all, dtype=np.int64)


def _tiled_predict(
    model: YOLO,
    image: np.ndarray,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    tile_w: int,
    tile_h: int,
    overlap: float,
    tile_batch: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    tile_w = int(min(max(1, tile_w), width))
    tile_h = int(min(max(1, tile_h), height))

    xs = _tile_starts(width, tile_w, overlap)
    ys = _tile_starts(height, tile_h, overlap)

    tiles: list[np.ndarray] = []
    offsets: list[Tuple[int, int]] = []
    for y in ys:
        for x in xs:
            tiles.append(image[y : y + tile_h, x : x + tile_w])
            offsets.append((x, y))

    if not tiles:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    predict_kwargs = {"imgsz": imgsz, "conf": conf, "iou": iou, "save": False, "verbose": False}
    if device:
        predict_kwargs["device"] = device

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []

    tile_batch = int(max(1, tile_batch))
    for start in range(0, len(tiles), tile_batch):
        batch_tiles = tiles[start : start + tile_batch]
        batch_offsets = offsets[start : start + tile_batch]
        batch_results = model.predict(source=batch_tiles, **predict_kwargs)
        for result, (x0, y0) in zip(batch_results, batch_offsets):
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy
            confs = boxes.conf
            cls = boxes.cls

            if hasattr(xyxy, "detach"):
                xyxy = xyxy.detach()
            if hasattr(confs, "detach"):
                confs = confs.detach()
            if hasattr(cls, "detach"):
                cls = cls.detach()

            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu()
            if hasattr(confs, "cpu"):
                confs = confs.cpu()
            if hasattr(cls, "cpu"):
                cls = cls.cpu()

            xyxy = xyxy.numpy() if hasattr(xyxy, "numpy") else np.asarray(xyxy)
            confs = confs.numpy() if hasattr(confs, "numpy") else np.asarray(confs)
            cls = cls.numpy() if hasattr(cls, "numpy") else np.asarray(cls)
            cls = cls.astype(np.int64, copy=False)

            xyxy = xyxy.astype(np.float32, copy=False)
            xyxy[:, [0, 2]] += float(x0)
            xyxy[:, [1, 3]] += float(y0)

            all_boxes.append(xyxy)
            all_scores.append(confs.astype(np.float32, copy=False))
            all_classes.append(cls)

    if not all_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)

    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)

    keep = _nms_by_class(boxes, scores, classes, iou_thr=iou)
    if keep.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return boxes[keep], scores[keep], classes[keep]


def _save_txt_yolo(
    txt_path: Path,
    *,
    image_shape: Tuple[int, int],
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    confs: Optional[np.ndarray] = None,
    save_conf: bool = False,
) -> None:
    height, width = image_shape[:2]
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box[:4].astype(float)
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 0 or bh <= 0:
            continue
        xc = (x1 + x2) / 2.0 / float(width)
        yc = (y1 + y2) / 2.0 / float(height)
        wn = bw / float(width)
        hn = bh / float(height)
        cls_id = int(classes[i]) if i < len(classes) else 0
        if save_conf and confs is not None and i < len(confs):
            lines.append(
                f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f} {float(confs[i]):.6f}"
            )
        else:
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _save_crops(
    crops_dir: Path,
    *,
    stem: str,
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    names,
) -> None:
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box[:4]
        x1 = int(max(0, round(float(x1))))
        y1 = int(max(0, round(float(y1))))
        x2 = int(min(image.shape[1], round(float(x2))))
        y2 = int(min(image.shape[0], round(float(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        cls_id = int(classes[i]) if i < len(classes) else 0
        cls_name = _class_name(names, cls_id).replace(" ", "_")
        out_dir = crops_dir / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        out_path = out_dir / f"{stem}_{i:03d}.jpg"
        cv2.imwrite(out_path.as_posix(), crop)


def _frame_stem(path_str: str, frame_index: int, is_video: bool) -> str:
    path = Path(path_str)
    stem = path.stem or "frame"
    if is_video:
        return f"{stem}_{frame_index + 1}"
    return stem


def _resolve_fps_for_source(source_fps: float, args) -> float:
    if source_fps and source_fps > 0:
        return float(source_fps)
    return float(args.video_fps)


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(model_path.as_posix())

    if args.tiled:
        tile_w = args.tile_w if args.tile_w is not None else args.imgsz
        tile_h = (
            args.tile_h if args.tile_h is not None else int(round(args.imgsz * 0.75))
        )
        overlap = float(args.tile_overlap)
        tile_batch = int(args.tile_batch)

        frames_iter, is_video, source_fps = _iter_source_frames(args.source)
        fps = _resolve_fps_for_source(source_fps, args)

        writer = None
        output_path = None
        frames_dir = None
        frame_index = 0
        visualize = args.save or args.show or args.keep_frames
        export_results = args.save_txt or args.save_crop
        save_dir = Path(args.project) / args.name

        if visualize or export_results:
            save_dir.mkdir(parents=True, exist_ok=True)

        try:
            for frame, path_str in frames_iter:
                boxes_xyxy, confs, classes = _tiled_predict(
                    model,
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    tile_w=tile_w,
                    tile_h=tile_h,
                    overlap=overlap,
                    tile_batch=tile_batch,
                )

                result = _FrameResult(
                    orig_img=frame,
                    path=path_str,
                    names=getattr(model, "names", None),
                    boxes=_BoxesView(boxes_xyxy, classes, confs),
                )
                stem = _frame_stem(path_str, frame_index, is_video)

                if export_results:
                    if args.save_txt:
                        txt_path = save_dir / "labels" / f"{stem}.txt"
                        _save_txt_yolo(
                            txt_path,
                            image_shape=frame.shape,
                            boxes_xyxy=boxes_xyxy,
                            classes=classes,
                            confs=confs,
                            save_conf=args.save_conf,
                        )
                    if args.save_crop:
                        _save_crops(
                            save_dir / "crops",
                            stem=stem,
                            image=frame,
                            boxes_xyxy=boxes_xyxy,
                            classes=classes,
                            names=getattr(model, "names", None),
                        )

                if visualize:
                    annotated = _draw_detections(frame.copy(), result)

                    if args.show:
                        cv2.imshow("predictions", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    if args.save:
                        if writer is None:
                            output_path = save_dir / "predictions.mp4"
                            writer = _init_video_writer(output_path, fps, annotated.shape)
                            if args.keep_frames:
                                frames_dir = save_dir / "frames"
                                frames_dir.mkdir(parents=True, exist_ok=True)
                        writer.write(annotated)

                    if args.keep_frames:
                        if frames_dir is None:
                            frames_dir = save_dir / "frames"
                            frames_dir.mkdir(parents=True, exist_ok=True)
                        frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
                        cv2.imwrite(frame_path.as_posix(), annotated)

                frame_index += 1
        finally:
            if writer is not None:
                writer.release()
            if args.show:
                cv2.destroyAllWindows()

        if args.save and output_path is not None:
            print(f"Saved video to {output_path}")
        return

    predict_kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "project": args.project,
        "name": args.name,
        "save": False,
        "show": False,
        "save_txt": False,
        "save_conf": False,
        "save_crop": False,
        "stream": True,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)
    writer = None
    output_path = None
    frames_dir = None
    save_dir = None
    frame_index = 0
    visualize = args.save or args.show or args.keep_frames
    export_results = args.save_txt or args.save_crop

    try:
        for result in results:
            if export_results:
                if save_dir is None:
                    save_dir = _resolve_save_dir(model, args, result)
                stem = _result_stem(result, model, frame_index)
                if args.save_txt:
                    txt_path = save_dir / "labels" / f"{stem}.txt"
                    result.save_txt(txt_path, save_conf=args.save_conf)
                if args.save_crop:
                    result.save_crop(save_dir=save_dir / "crops", file_name=stem)

            if visualize:
                if result.orig_img is None:
                    frame_index += 1
                    continue
                frame = result.orig_img.copy()
                annotated = _draw_detections(frame, result)

                if args.show:
                    cv2.imshow("predictions", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.save:
                    if writer is None:
                        if save_dir is None:
                            save_dir = _resolve_save_dir(model, args, result)
                        fps = _resolve_fps(model, args)
                        output_path = save_dir / "predictions.mp4"
                        writer = _init_video_writer(output_path, fps, annotated.shape)
                        if args.keep_frames:
                            frames_dir = save_dir / "frames"
                            frames_dir.mkdir(parents=True, exist_ok=True)
                    writer.write(annotated)

                if args.keep_frames:
                    if frames_dir is None:
                        if save_dir is None:
                            save_dir = _resolve_save_dir(model, args, result)
                        frames_dir = save_dir / "frames"
                        frames_dir.mkdir(parents=True, exist_ok=True)
                    frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
                    cv2.imwrite(frame_path.as_posix(), annotated)

            frame_index += 1
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if args.save and output_path is not None:
        print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
