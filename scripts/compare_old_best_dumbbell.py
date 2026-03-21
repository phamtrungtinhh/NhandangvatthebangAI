#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict

from ultralytics import YOLO

DATASET_ROOT = Path("S:/workspace/model_custom/dataset")
SAMPLE_JSON = Path("S:/workspace/tmp_test_outputs/dumbbell_infer_best.json")
OLD_WEIGHTS = Path("S:/workspace/model_custom/weights/custo_all.pt")
BEST_WEIGHTS = Path("S:/workspace/model_custom/_train_runs/dumbbell_merge_20260314/weights/best.pt")
OUT_JSON = Path("S:/workspace/tmp_test_outputs/dumbbell_old_vs_best_compare.json")
IOU_THR = 0.5
CONF_THR = 0.25


def xywhn_to_xyxy(x: float, y: float, w: float, h: float, iw: int, ih: int) -> Tuple[float, float, float, float]:
    bw = w * iw
    bh = h * ih
    cx = x * iw
    cy = y * ih
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return x1, y1, x2, y2


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_sample_images() -> List[Path]:
    payload = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))
    return [Path(item["image"]) for item in payload.get("samples", [])]


def find_split_and_label(img_path: Path) -> Path:
    split = img_path.parent.parent.name
    label_path = DATASET_ROOT / split / "labels" / f"{img_path.stem}.txt"
    return label_path


def load_gt_class0_boxes(img_path: Path) -> List[Tuple[float, float, float, float]]:
    label_path = find_split_and_label(img_path)
    if not label_path.exists():
        return []
    from PIL import Image

    with Image.open(img_path) as im:
        iw, ih = im.size

    out: List[Tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
        except Exception:
            continue
        if cls_id != 0:
            continue
        out.append(xywhn_to_xyxy(x, y, w, h, iw, ih))
    return out


def predict_class0_boxes(model: YOLO, img_path: Path) -> List[Tuple[float, float, float, float, float]]:
    r = model.predict(source=str(img_path), imgsz=640, conf=CONF_THR, max_det=100, verbose=False)[0]
    out: List[Tuple[float, float, float, float, float]] = []
    boxes = r.boxes
    if boxes is None or boxes.cls is None:
        return out
    cls_list = boxes.cls.tolist()
    conf_list = boxes.conf.tolist()
    xyxy_list = boxes.xyxy.tolist()
    for i, c in enumerate(cls_list):
        if int(c) != 0:
            continue
        x1, y1, x2, y2 = xyxy_list[i]
        out.append((x1, y1, x2, y2, float(conf_list[i])))
    return out


def match_counts(gt_boxes: List[Tuple[float, float, float, float]], pred_boxes: List[Tuple[float, float, float, float, float]]) -> Dict[str, int]:
    used_gt = set()
    used_pr = set()

    # Greedy matching by IoU
    pairs = []
    for pi, p in enumerate(pred_boxes):
        pb = (p[0], p[1], p[2], p[3])
        for gi, g in enumerate(gt_boxes):
            pairs.append((iou(pb, g), pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    tp = 0
    for score, pi, gi in pairs:
        if score < IOU_THR:
            break
        if pi in used_pr or gi in used_gt:
            continue
        used_pr.add(pi)
        used_gt.add(gi)
        tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return {"tp": tp, "fp": fp, "fn": fn}


def eval_model(model_path: Path, images: List[Path]) -> Dict[str, object]:
    model = YOLO(str(model_path))
    total_tp = total_fp = total_fn = 0
    img_with_pred = 0
    per_image = []

    for img in images:
        gt = load_gt_class0_boxes(img)
        pr = predict_class0_boxes(model, img)
        c = match_counts(gt, pr)
        total_tp += c["tp"]
        total_fp += c["fp"]
        total_fn += c["fn"]
        if len(pr) > 0:
            img_with_pred += 1
        per_image.append(
            {
                "image": str(img),
                "gt_class0": len(gt),
                "pred_class0": len(pr),
                **c,
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

    return {
        "weights": str(model_path),
        "images": len(images),
        "iou_thr": IOU_THR,
        "conf_thr": CONF_THR,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "images_with_pred_class0": img_with_pred,
        "per_image": per_image,
    }


def main() -> int:
    images = load_sample_images()
    old_metrics = eval_model(OLD_WEIGHTS, images)
    best_metrics = eval_model(BEST_WEIGHTS, images)

    payload = {
        "sample_source": str(SAMPLE_JSON),
        "images": [str(p) for p in images],
        "old": old_metrics,
        "best": best_metrics,
        "delta": {
            "precision": best_metrics["precision"] - old_metrics["precision"],
            "recall": best_metrics["recall"] - old_metrics["recall"],
            "tp": best_metrics["tp"] - old_metrics["tp"],
            "fp": best_metrics["fp"] - old_metrics["fp"],
            "fn": best_metrics["fn"] - old_metrics["fn"],
            "images_with_pred_class0": best_metrics["images_with_pred_class0"] - old_metrics["images_with_pred_class0"],
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Compared on", len(images), "images")
    print("old precision", f"{old_metrics['precision']:.4f}", "recall", f"{old_metrics['recall']:.4f}", "tp/fp/fn", old_metrics['tp'], old_metrics['fp'], old_metrics['fn'])
    print("best precision", f"{best_metrics['precision']:.4f}", "recall", f"{best_metrics['recall']:.4f}", "tp/fp/fn", best_metrics['tp'], best_metrics['fp'], best_metrics['fn'])
    print("delta precision", f"{payload['delta']['precision']:.4f}", "delta recall", f"{payload['delta']['recall']:.4f}")
    print("report", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
