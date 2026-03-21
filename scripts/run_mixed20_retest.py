#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
import importlib.util

from PIL import Image
from ultralytics import YOLO

spec = importlib.util.spec_from_file_location("app", "S:/workspace/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

DATASET = Path("S:/workspace/model_custom/dataset")
OUT_DIR = Path("S:/workspace/tmp_test_outputs/flower_rule_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = Path("S:/workspace/model_custom/weights/custo_all.pt")
CUSTOM_MODEL = YOLO(str(MODEL))
COCO_MODEL = YOLO(str(app.COCO_MODEL_PATH))

DEVICE = os.environ.get("FLOWER_RETEST_DEVICE", "0")
SEED = int(os.environ.get("FLOWER_RETEST_MIXED_SEED", "20260316"))
TARGET_SIZE = 20
TARGET_CLASSES = ["flower", "tree", "fruit", "dumbbell"]
GT_ID_TO_NAME = {0: "dumbbell", 1: "flower", 2: "fruit", 3: "tree"}


def counts_from_detections(dets: list[dict]) -> dict[str, int]:
    c = Counter()
    for d in dets or []:
        cls_name = app.normalize_class_name(d.get("class", "")) or "unknown"
        c[cls_name] += 1
    return dict(c)


def parse_label_counts(label_path: Path) -> dict[str, int]:
    counts = Counter()
    for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = ln.strip().split()
        if len(p) < 5:
            continue
        try:
            cls_id = int(float(p[0]))
        except Exception:
            continue
        name = GT_ID_TO_NAME.get(cls_id)
        if name:
            counts[name] += 1
    return dict(counts)


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
        c = images_dir / f"{stem}{ext}"
        if c.exists():
            return c
    return None


def collect_candidates() -> list[dict]:
    out = []
    for split in ["train", "val"]:
        labels_dir = DATASET / split / "labels"
        images_dir = DATASET / split / "images"
        if not labels_dir.exists() or not images_dir.exists():
            continue
        for lbl in labels_dir.glob("*.txt"):
            gt_counts = parse_label_counts(lbl)
            if not any(gt_counts.get(k, 0) > 0 for k in TARGET_CLASSES):
                continue
            img = find_image_for_label(images_dir, lbl.stem)
            if img is None:
                continue
            present = [k for k in TARGET_CLASSES if gt_counts.get(k, 0) > 0]
            out.append(
                {
                    "image": str(img),
                    "split": split,
                    "gt_counts": gt_counts,
                    "present_classes": present,
                }
            )
    return out


def select_mixed20(candidates: list[dict]) -> list[dict]:
    random.seed(SEED)
    random.shuffle(candidates)

    selected = []
    used = set()
    target_per_class = max(1, TARGET_SIZE // len(TARGET_CLASSES))

    # Ensure each class appears enough times.
    for cls in TARGET_CLASSES:
        cls_pool = [x for x in candidates if cls in x["present_classes"] and x["image"] not in used]
        take = min(target_per_class, len(cls_pool))
        for x in cls_pool[:take]:
            selected.append(x)
            used.add(x["image"])

    # Fill remaining slots with images that increase class diversity.
    remaining = [x for x in candidates if x["image"] not in used]
    remaining.sort(key=lambda x: len(x.get("present_classes", [])), reverse=True)
    for x in remaining:
        if len(selected) >= TARGET_SIZE:
            break
        selected.append(x)
        used.add(x["image"])

    return selected[:TARGET_SIZE]


def run_hybrid_pipeline(img_path: Path) -> dict:
    res_c = CUSTOM_MODEL.predict(source=str(img_path), conf=0.20, imgsz=512, device=DEVICE, verbose=False)
    r_c = res_c[0]
    custom_dets = []
    if getattr(r_c, "boxes", None) is not None and len(r_c.boxes) > 0:
        cls_list = r_c.boxes.cls.tolist()
        conf_list = r_c.boxes.conf.tolist()
        xyxy_list = [list(map(float, b.xyxy[0].cpu().numpy())) if hasattr(b, "xyxy") else None for b in r_c.boxes]
        for idx, (cls, conf) in enumerate(zip(cls_list, conf_list)):
            cname = str(CUSTOM_MODEL.model.names[int(cls)])
            xy = xyxy_list[idx] if idx < len(xyxy_list) else None
            custom_dets.append({"class": cname, "conf": float(conf), "xyxy": xy, "_source_model": "custom"})

    res_co = COCO_MODEL.predict(source=str(img_path), conf=0.20, imgsz=512, device=DEVICE, verbose=False)
    r_co = res_co[0]
    coco_dets = []
    if getattr(r_co, "boxes", None) is not None and len(r_co.boxes) > 0:
        cls_list = r_co.boxes.cls.tolist()
        conf_list = r_co.boxes.conf.tolist()
        xyxy_list = [list(map(float, b.xyxy[0].cpu().numpy())) if hasattr(b, "xyxy") else None for b in r_co.boxes]
        for idx, (cls, conf) in enumerate(zip(cls_list, conf_list)):
            cname = str(COCO_MODEL.model.names[int(cls)])
            xy = xyxy_list[idx] if idx < len(xyxy_list) else None
            coco_dets.append({"class": cname, "conf": float(conf), "xyxy": xy, "_source_model": "coco"})

    merged = custom_dets + coco_dets
    image_pil = Image.open(img_path).convert("RGB")
    w, h = image_pil.size
    image_np = None
    try:
        import numpy as np

        image_np = np.array(image_pil)
    except Exception:
        pass

    merged = app.refine_custom_detections_with_specialists(image_np, merged, base_conf=0.35, imgsz=640, use_fp16=True)
    after_verify = app.verify_and_reduce_detections(merged, img_w=w, img_h=h, base_conf=0.35)
    ret = app.finalize_frame_detections_for_count(after_verify, img_w=w, img_h=h, min_conf=0.35, debug=True)
    final, stage_stats, stage_dets = ret if isinstance(ret, tuple) and len(ret) == 3 else (ret, {}, {})

    return {
        "after_finalize_counts": counts_from_detections(final or []),
        "stage_stats": stage_stats or {},
        "stage_detections": stage_dets or {},
    }


def total_abs_error(gt_counts: dict, pred_counts: dict) -> int:
    return sum(abs(int(pred_counts.get(c, 0)) - int(gt_counts.get(c, 0))) for c in TARGET_CLASSES)


def per_image_mae(gt_counts: dict, pred_counts: dict) -> float:
    return float(total_abs_error(gt_counts, pred_counts)) / float(len(TARGET_CLASSES))


def dominant_drop(stage_stats: dict) -> tuple[str | None, int]:
    order = [
        "after_nms",
        "after_resolve_cross_class",
        "after_cross_class_suppress",
        "after_generic_fruit_overlap",
        "after_specific_fruit_conflict",
        "after_scene_level",
        "after_one_object_box",
        "after_dense_normalize",
        "after_flower_pruning",
        "after_scene_rules",
        "final",
    ]
    max_drop = 0
    max_stage = None
    prev = None
    for s in order:
        cur = int(stage_stats.get(s, 0) or 0)
        if prev is not None:
            drop = prev - cur
            if drop > max_drop:
                max_drop = drop
                max_stage = s
        prev = cur
    return max_stage, max_drop


def make_summary(items: list[dict], before: dict | None = None) -> dict:
    per_class_total = Counter()
    stage_drop_acc = Counter()
    fallback_count = 0

    for it in items:
        for c in TARGET_CLASSES:
            per_class_total[c] += int(it.get("pred_counts", {}).get(c, 0))
        if bool(it.get("stage_stats", {}).get("fallback_used")):
            fallback_count += 1
        ds = it.get("dominant_drop_stage")
        dv = int(it.get("dominant_drop_value") or 0)
        if ds and dv > 0:
            stage_drop_acc[ds] += dv

    mae_after = sum(float(it.get("mae", 0.0)) for it in items) / max(1, len(items))

    mae_before = None
    improved = None
    worsened = None
    same = None
    if before and isinstance(before, dict):
        before_items = before.get("items", [])
        bmap = {str(x.get("image")): x for x in before_items}
        common = [it for it in items if str(it.get("image")) in bmap]
        if common:
            deltas = []
            for it in common:
                b = bmap[str(it.get("image"))]
                ma = float(it.get("mae", 0.0))
                mb = float(b.get("mae", 0.0))
                deltas.append((mb, ma))
            mae_before = sum(mb for mb, _ in deltas) / max(1, len(deltas))
            improved = sum(1 for mb, ma in deltas if ma < mb)
            worsened = sum(1 for mb, ma in deltas if ma > mb)
            same = sum(1 for mb, ma in deltas if ma == mb)

    top_drop_stages = [{"stage": k, "drop_sum": v} for k, v in stage_drop_acc.most_common()]

    return {
        "mixed20_size": len(items),
        "mae_before": mae_before,
        "mae_after": mae_after,
        "improved": improved,
        "worsened": worsened,
        "same": same,
        "per_class_counts_total": dict(per_class_total),
        "top_drop_stages": top_drop_stages,
        "fallback_count": fallback_count,
        "items": items,
    }


def main() -> int:
    before_path = OUT_DIR / "mixed20_before_snapshot.json"
    after_path = OUT_DIR / "mixed20_after_summary.json"

    # If no explicit before snapshot but previous after exists, freeze it as before.
    if (not before_path.exists()) and after_path.exists():
        before_path.write_text(after_path.read_text(encoding="utf-8"), encoding="utf-8")

    before = None
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding="utf-8"))

    candidates = collect_candidates()
    selected = select_mixed20(candidates)

    items = []
    for it in selected:
        img = Path(it["image"])
        res = run_hybrid_pipeline(img)
        pred_counts = res.get("after_finalize_counts") or {}
        gt_counts = it.get("gt_counts") or {}
        mae = per_image_mae(gt_counts, pred_counts)
        err_by_class = {
            c: int(pred_counts.get(c, 0)) - int(gt_counts.get(c, 0))
            for c in TARGET_CLASSES
        }
        drop_stage, drop_val = dominant_drop(res.get("stage_stats") or {})
        items.append(
            {
                "image": str(img),
                "split": it.get("split"),
                "present_classes": it.get("present_classes") or [],
                "gt_counts": {c: int(gt_counts.get(c, 0)) for c in TARGET_CLASSES},
                "pred_counts": {c: int(pred_counts.get(c, 0)) for c in TARGET_CLASSES},
                "error_by_class": err_by_class,
                "mae": float(mae),
                "stage_stats": res.get("stage_stats") or {},
                "dominant_drop_stage": drop_stage,
                "dominant_drop_value": int(drop_val),
            }
        )

    summary = make_summary(items, before=before)
    after_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("wrote", after_path)
    print(json.dumps({
        "mixed20_size": summary.get("mixed20_size"),
        "mae_before": summary.get("mae_before"),
        "mae_after": summary.get("mae_after"),
        "improved": summary.get("improved"),
        "worsened": summary.get("worsened"),
        "same": summary.get("same"),
        "per_class_counts_total": summary.get("per_class_counts_total"),
        "top_drop_stages": summary.get("top_drop_stages", [])[:5],
        "fallback_count": summary.get("fallback_count"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
