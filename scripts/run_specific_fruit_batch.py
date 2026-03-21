#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
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
OUT_JSON = OUT_DIR / "specific_fruit_batch_report.json"

DEVICE = os.environ.get("FLOWER_RETEST_DEVICE", "cpu")
TARGET = 20
SEED = 20260316
SPECIFIC = {"orange", "banana", "apple"}

custom_model = YOLO("S:/workspace/model_custom/weights/custo_all.pt")
coco_model = YOLO(str(app.COCO_MODEL_PATH))


def iou(a, b):
    return app._box_iou(a, b)


def normalize(dets):
    return app.canonicalize_final_detections(dets)


def run_models(img_path: Path):
    res_c = custom_model.predict(source=str(img_path), conf=0.20, imgsz=512, device=DEVICE, verbose=False)
    r_c = res_c[0]
    custom_dets = []
    if getattr(r_c, "boxes", None) is not None and len(r_c.boxes) > 0:
        cls_list = r_c.boxes.cls.tolist()
        conf_list = r_c.boxes.conf.tolist()
        xyxy_list = [list(map(float, b.xyxy[0].cpu().numpy())) if hasattr(b, "xyxy") else None for b in r_c.boxes]
        for idx, (cls, conf) in enumerate(zip(cls_list, conf_list)):
            cname = str(custom_model.model.names[int(cls)])
            xy = xyxy_list[idx] if idx < len(xyxy_list) else None
            custom_dets.append({"class": cname, "conf": float(conf), "xyxy": xy, "_source_model": "custom"})

    res_co = coco_model.predict(source=str(img_path), conf=0.20, imgsz=512, device=DEVICE, verbose=False)
    r_co = res_co[0]
    coco_dets = []
    if getattr(r_co, "boxes", None) is not None and len(r_co.boxes) > 0:
        cls_list = r_co.boxes.cls.tolist()
        conf_list = r_co.boxes.conf.tolist()
        xyxy_list = [list(map(float, b.xyxy[0].cpu().numpy())) if hasattr(b, "xyxy") else None for b in r_co.boxes]
        for idx, (cls, conf) in enumerate(zip(cls_list, conf_list)):
            cname = str(coco_model.model.names[int(cls)])
            xy = xyxy_list[idx] if idx < len(xyxy_list) else None
            coco_dets.append({"class": cname, "conf": float(conf), "xyxy": xy, "_source_model": "coco"})

    return custom_dets, coco_dets


def has_specific(coco_dets):
    for d in coco_dets:
        c = app.normalize_class_name(d.get("class", "")) or ""
        if c in SPECIFIC:
            return True
    return False


def counts(dets):
    out = {}
    for d in dets:
        c = app.normalize_class_name(d.get("class", "")) or "unknown"
        out[c] = int(out.get(c, 0)) + 1
    return out


def overlap_duplicate_count(final_dets):
    fruits = [d for d in final_dets if (app.normalize_class_name(d.get("class", "")) or "") == "fruit"]
    specs = [d for d in final_dets if (app.normalize_class_name(d.get("class", "")) or "") in SPECIFIC]
    dup = 0
    for f in fruits:
        if any(iou(f.get("xyxy"), s.get("xyxy")) >= 0.35 for s in specs):
            dup += 1
    return dup


def dataset_images():
    imgs = []
    for split in ("train", "val"):
        p = DATASET / split / "images"
        if not p.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            imgs.extend(p.glob(ext))
    random.seed(SEED)
    random.shuffle(imgs)
    return imgs


def main():
    chosen = []
    items = []
    total_pred = {"orange": 0, "banana": 0, "apple": 0, "fruit": 0}
    total_dup = 0

    for img in dataset_images():
        if len(chosen) >= TARGET:
            break
        custom_dets, coco_dets = run_models(img)
        if not has_specific(coco_dets):
            continue

        merged = normalize(custom_dets + coco_dets)
        w, h = Image.open(img).convert("RGB").size
        after_verify = app.verify_and_reduce_detections(merged, img_w=w, img_h=h, base_conf=0.35)
        ret = app.finalize_frame_detections_for_count(after_verify, img_w=w, img_h=h, min_conf=0.35, debug=True)
        final_dets, stage_stats, _ = ret if isinstance(ret, tuple) and len(ret) == 3 else (ret, {}, {})
        final_dets = normalize(final_dets or [])

        c = counts(final_dets)
        dup = overlap_duplicate_count(final_dets)
        total_dup += dup
        for k in total_pred.keys():
            total_pred[k] += int(c.get(k, 0))

        items.append(
            {
                "image": str(img),
                "final_counts": {k: int(c.get(k, 0)) for k in ["orange", "banana", "apple", "fruit"]},
                "overlap_fruit_with_specific": int(dup),
                "fallback_used": bool((stage_stats or {}).get("fallback_used")),
            }
        )
        chosen.append(img)

    summary = {
        "batch_size": len(items),
        "device": DEVICE,
        "total_final_counts": total_pred,
        "images_with_overlap_duplicate": int(sum(1 for it in items if it.get("overlap_fruit_with_specific", 0) > 0)),
        "overlap_duplicate_total": int(total_dup),
        "fallback_count": int(sum(1 for it in items if it.get("fallback_used"))),
        "items": items,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT_JSON)
    print(json.dumps({
        "batch_size": summary["batch_size"],
        "total_final_counts": summary["total_final_counts"],
        "images_with_overlap_duplicate": summary["images_with_overlap_duplicate"],
        "overlap_duplicate_total": summary["overlap_duplicate_total"],
        "fallback_count": summary["fallback_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
