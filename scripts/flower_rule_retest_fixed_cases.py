#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import importlib.util
import numpy as np
from ultralytics import YOLO

# Load app.py explicitly
spec = importlib.util.spec_from_file_location("app", "S:/workspace/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

MODEL = Path("S:/workspace/model_custom/weights/custo_all.pt")
custom_model = YOLO(str(MODEL))
coco_model = YOLO(str(app.COCO_MODEL_PATH))
DEVICE = os.environ.get("FLOWER_RETEST_DEVICE", "0")

IN_JSON = Path("S:/workspace/tmp_test_outputs/flower_rule_debug/flower_debug_report_before_rule1.json")
OUT_JSON = Path("S:/workspace/tmp_test_outputs/flower_rule_debug/flower_debug_report_after_rule1_on_fixed_cases.json")


def counts_from(dets: list[dict[str, Any]] | None) -> dict[str, int]:
    c: dict[str, int] = {}
    for d in dets or []:
        k = app.normalize_class_name(d.get("class", "")) or "unknown"
        c[k] = c.get(k, 0) + 1
    return c


def flower_count_from_stage_dets(stage_dets: dict[str, list[dict[str, Any]]] | None) -> dict[str, int]:
    out = {}
    for k, ds in (stage_dets or {}).items():
        cnt = 0
        for d in ds or []:
            if app.normalize_class_name(d.get("class", "")) == "flower":
                cnt += 1
        out[k] = cnt
    return out


def dominant_drop_stage(flower_stage_counts: dict[str, int]) -> tuple[str | None, int]:
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


def run_hybrid_pipeline(img_path: Path) -> dict[str, Any]:
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
    image_np = np.array(image_pil)

    # Keep specialist usage aligned with app runtime behavior.
    merged = app.refine_custom_detections_with_specialists(
        image_np,
        merged,
        base_conf=0.35,
        imgsz=640,
        use_fp16=True,
    )
    after_verify = app.verify_and_reduce_detections(merged, img_w=w, img_h=h, base_conf=0.35)

    ret = app.finalize_frame_detections_for_count(after_verify, img_w=w, img_h=h, min_conf=0.35, debug=True)
    final, stage_stats, stage_dets = ret

    return {
        "after_finalize_counts": counts_from(final),
        "stage_stats": stage_stats,
        "stage_detections": stage_dets,
    }


def main() -> int:
    before = json.loads(IN_JSON.read_text(encoding="utf-8"))
    items_in = before.get("items", [])

    stage_drop_acc = defaultdict(int)
    items_out = []

    for it in items_in:
        img = Path(it["image"])
        if not img.exists():
            continue
        res = run_hybrid_pipeline(img)
        flower_stage_counts = flower_count_from_stage_dets(res.get("stage_detections") or {})
        drop_stage, drop_val = dominant_drop_stage(flower_stage_counts)
        if drop_stage and drop_val > 0:
            stage_drop_acc[drop_stage] += drop_val

        gt = it.get("gt_flower")
        final_flower = int((res.get("after_finalize_counts") or {}).get("flower", 0))
        err = None
        if gt is not None:
            err = int(final_flower - int(gt))

        items_out.append(
            {
                "image": str(img),
                "source": it.get("source"),
                "split": it.get("split"),
                "gt_flower": gt,
                "final_flower": final_flower,
                "error": err,
                "stage_stats": res.get("stage_stats") or {},
                "flower_stage_counts": flower_stage_counts,
                "dominant_drop_stage": drop_stage,
                "dominant_drop_value": drop_val,
            }
        )

    top_drop_stages = sorted(stage_drop_acc.items(), key=lambda x: x[1], reverse=True)
    out = {
        "summary": {
            "selected_total": len(items_out),
            "dataset_cases": len([x for x in items_out if x.get("source") == "dataset"]),
            "real_cases": len([x for x in items_out if x.get("source") == "real"]),
        },
        "top_drop_stages": [{"stage": k, "drop_sum": v} for k, v in top_drop_stages],
        "items": items_out,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("selected_total", out["summary"]["selected_total"])
    print("top_drop_stages", out["top_drop_stages"][:5])
    print("report", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
