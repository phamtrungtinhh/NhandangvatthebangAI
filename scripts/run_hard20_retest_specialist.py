#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
import importlib.util

# load app
spec = importlib.util.spec_from_file_location("app", "S:/workspace/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

from ultralytics import YOLO

DATASET = Path("S:/workspace/model_custom/dataset")
OUT_DIR = Path("S:/workspace/tmp_test_outputs/flower_rule_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = Path("S:/workspace/model_custom/weights/custo_all.pt")
custom_model = YOLO(str(MODEL))
coco_model = YOLO(str(app.COCO_MODEL_PATH))

DEVICE = os.environ.get("FLOWER_RETEST_DEVICE", "0")


def image_candidates_from_dataset() -> list[dict]:
    out = []
    for split in ["train", "val"]:
        labels_dir = DATASET / split / "labels"
        images_dir = DATASET / split / "images"
        if not labels_dir.exists() or not images_dir.exists():
            continue
        for lbl in labels_dir.glob("*.txt"):
            gt_flower = 0
            for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = ln.strip().split()
                if len(p) < 5:
                    continue
                try:
                    cls = int(float(p[0]))
                except Exception:
                    continue
                if cls == 1:
                    gt_flower += 1
            if gt_flower <= 0:
                continue
            img = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                c = images_dir / f"{lbl.stem}{ext}"
                if c.exists():
                    img = c
                    break
            if img is None:
                continue
            out.append({"image": str(img), "gt_flower": gt_flower, "split": split})
    return out


def run_hybrid_pipeline(img_path: Path) -> dict:
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

    merged = custom_dets + coco_dets
    from PIL import Image
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

    final_counts = {}
    for d in final or []:
        cls_name = app.normalize_class_name(d.get("class", "")) or "unknown"
        final_counts[cls_name] = int(final_counts.get(cls_name, 0)) + 1

    return {
        "after_finalize_counts": final_counts,
        "stage_stats": stage_stats,
        "stage_detections": stage_dets,
    }


def flower_count_from_stage_dets(stage_dets: dict) -> dict:
    out = {}
    for k, ds in (stage_dets or {}).items():
        cnt = 0
        for d in ds or []:
            if app.normalize_class_name(d.get("class", "")) == "flower":
                cnt += 1
        out[k] = cnt
    return out


def dominant_drop_stage(flower_stage_counts: dict) -> tuple:
    order = ["after_nms", "after_resolve_cross_class", "after_flower_pruning", "after_scene_rules", "final"]
    max_drop = 0
    max_stage = None
    prev = None
    for s in order:
        cur = int(flower_stage_counts.get(s, 0))
        if prev is not None:
            drop = prev - cur
            if drop > max_drop:
                max_drop = drop
                max_stage = s
        prev = cur
    return max_stage, max_drop


def main():
    cand = image_candidates_from_dataset()
    scored = []
    for it in cand:
        img = Path(it["image"]) if isinstance(it.get("image"), str) else Path(it.get("image"))
        # fast pass: run pipeline to get final counts
        try:
            res = run_hybrid_pipeline(img)
            final_flower = int((res.get("after_finalize_counts") or {}).get("flower", 0))
        except Exception:
            final_flower = 0
        gt = int(it.get("gt_flower") or 0)
        err = final_flower - gt
        scored.append({**it, "pred_flower": final_flower, "error": err, "abs_error": abs(err)})

    scored.sort(key=lambda x: x["abs_error"], reverse=True)
    hard20 = scored[:20]

    items = []
    stage_drop_acc = defaultdict(int)
    for it in hard20:
        img = Path(it["image"])
        res = run_hybrid_pipeline(img)
        stage_dets = res.get("stage_detections") or {}
        flower_stage_counts = flower_count_from_stage_dets(stage_dets)
        drop_stage, drop_val = dominant_drop_stage(flower_stage_counts)
        if drop_stage and drop_val > 0:
            stage_drop_acc[drop_stage] += drop_val

        gt = it.get("gt_flower")
        final_flower = int((res.get("after_finalize_counts") or {}).get("flower", 0))
        err = None
        if gt is not None:
            err = int(final_flower - int(gt))

        items.append({
            "image": str(img),
            "gt_flower": gt,
            "final_flower": final_flower,
            "final_class_counts": res.get("after_finalize_counts") or {},
            "error": err,
            "stage_stats": res.get("stage_stats") or {},
            "flower_stage_counts": flower_stage_counts,
            "dominant_drop_stage": drop_stage,
            "dominant_drop_value": drop_val,
        })

    top_drop_stages = sorted(stage_drop_acc.items(), key=lambda x: x[1], reverse=True)
    summary = {
        "hard20_size": len(items),
        "mae": sum(abs(x.get("error") or 0) for x in items) / max(1, len(items)),
        "top_drop_stages": [{"stage": k, "drop_sum": v} for k, v in top_drop_stages],
        "items": items,
    }
    out_path = OUT_DIR / "hard20_final_specialist_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out_path)


if __name__ == '__main__':
    main()
