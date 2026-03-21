#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import importlib.util
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Load app.py explicitly
spec = importlib.util.spec_from_file_location("app", "S:/workspace/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

DATASET = Path("S:/workspace/model_custom/dataset")
REAL_DIR = Path("S:/workspace/tmp_test_inputs/user_case_verify")
OUT_DIR = Path("S:/workspace/tmp_test_outputs/flower_rule_debug")
SNAP_DIR = OUT_DIR / "snapshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SNAP_DIR.mkdir(parents=True, exist_ok=True)

MODEL = Path("S:/workspace/model_custom/weights/custo_all.pt")
custom_model = YOLO(str(MODEL))
coco_model = YOLO(str(app.COCO_MODEL_PATH))

RANDOM_SEED = 42
TARGET_DATASET_CASES = 40


def image_candidates_from_dataset() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
                c = images_dir / f"{lbl.stem}{ext}"
                if c.exists():
                    img = c
                    break
            if img is None:
                continue
            out.append(
                {
                    "image": str(img),
                    "source": "dataset",
                    "split": split,
                    "gt_flower": gt_flower,
                }
            )
    return out


def real_candidates() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not REAL_DIR.exists():
        return out
    for p in sorted(REAL_DIR.glob("*")):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            out.append({"image": str(p), "source": "real", "split": None, "gt_flower": None})
    return out


def counts_from(dets: list[dict[str, Any]] | None) -> dict[str, int]:
    c: dict[str, int] = {}
    for d in dets or []:
        k = app.normalize_class_name(d.get("class", "")) or "unknown"
        c[k] = c.get(k, 0) + 1
    return c


def run_hybrid_pipeline(img_path: Path, debug: bool = True) -> dict[str, Any]:
    res_c = custom_model.predict(source=str(img_path), conf=0.20, imgsz=512, device=0, verbose=False)
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

    res_co = coco_model.predict(source=str(img_path), conf=0.20, imgsz=512, device=0, verbose=False)
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
    w, h = Image.open(img_path).size
    after_verify = app.verify_and_reduce_detections(merged, img_w=w, img_h=h, base_conf=0.35)

    stage_stats = {}
    stage_dets = {}
    try:
        ret = app.finalize_frame_detections_for_count(after_verify, img_w=w, img_h=h, min_conf=0.35, debug=debug)
        if isinstance(ret, tuple):
            if len(ret) == 3:
                final, stage_stats, stage_dets = ret
            elif len(ret) == 2:
                final, stage_stats = ret
                stage_dets = {}
            else:
                final = ret[0]
        else:
            final = ret
    except Exception:
        final = after_verify
        stage_stats = {}
        stage_dets = {}

    return {
        "raw_custom_counts": counts_from(custom_dets),
        "raw_coco_counts": counts_from(coco_dets),
        "after_verify_counts": counts_from(after_verify),
        "after_finalize_counts": counts_from(final),
        "stage_stats": stage_stats,
        "stage_detections_counts": {k: len(v) for k, v in (stage_dets or {}).items()},
        "stage_detections": stage_dets,
    }


def draw_stage_snapshot(img_path: Path, stage_name: str, dets: list[dict[str, Any]], out_path: Path) -> None:
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in dets or []:
        xy = d.get("xyxy") or []
        if len(xy) < 4:
            continue
        try:
            x1, y1, x2, y2 = map(float, xy[:4])
        except Exception:
            continue
        cls_name = app.normalize_class_name(d.get("class", "")) or "unknown"
        color = "yellow" if cls_name == "flower" else "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        txt = f"{cls_name}:{float(d.get('conf', 0.0)):.2f}"
        draw.text((x1 + 2, y1 + 2), txt, fill=color, font=font)
    draw.text((8, 8), f"{stage_name}", fill="white", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


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
    order = [
        "after_nms",
        "after_resolve_cross_class",
        "after_flower_pruning",
        "after_scene_rules",
        "final",
    ]
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


def main() -> int:
    random.seed(RANDOM_SEED)
    dataset_cases = image_candidates_from_dataset()

    # First pass: compute prediction error on dataset to pick hard cases.
    scored = []
    for it in dataset_cases:
        img = Path(it["image"])
        res = run_hybrid_pipeline(img, debug=False)
        pred_flower = int((res.get("after_finalize_counts") or {}).get("flower", 0))
        gt_flower = int(it.get("gt_flower") or 0)
        err = pred_flower - gt_flower
        scored.append({**it, "pred_flower": pred_flower, "error": err, "abs_error": abs(err)})

    scored.sort(key=lambda x: x["abs_error"], reverse=True)

    # Ensure miss + overcount coverage
    under = [x for x in scored if x["error"] < 0][: TARGET_DATASET_CASES // 2]
    over = [x for x in scored if x["error"] > 0][: TARGET_DATASET_CASES // 2]
    chosen = under + over
    if len(chosen) < TARGET_DATASET_CASES:
        used = {x["image"] for x in chosen}
        for x in scored:
            if x["image"] in used:
                continue
            chosen.append(x)
            if len(chosen) >= TARGET_DATASET_CASES:
                break

    real = real_candidates()
    selected = chosen + real  # typically 40 + 6 = 46

    items = []
    stage_drop_acc = defaultdict(int)

    for idx, it in enumerate(selected, start=1):
        img = Path(it["image"])
        res = run_hybrid_pipeline(img, debug=True)
        stage_dets = res.get("stage_detections") or {}
        flower_stage_counts = flower_count_from_stage_dets(stage_dets)
        drop_stage, drop_val = dominant_drop_stage(flower_stage_counts)
        if drop_stage and drop_val > 0:
            stage_drop_acc[drop_stage] += drop_val

        # Save snapshots for key stages.
        for s in ["after_nms", "after_flower_pruning", "after_scene_rules", "final"]:
            dets = stage_dets.get(s)
            if dets is None:
                continue
            out_img = SNAP_DIR / f"{idx:03d}_{img.stem}__{s}.jpg"
            draw_stage_snapshot(img, s, dets, out_img)

        item = {
            "image": str(img),
            "source": it.get("source"),
            "split": it.get("split"),
            "gt_flower": it.get("gt_flower"),
            "raw_custom_flower": int((res.get("raw_custom_counts") or {}).get("flower", 0)),
            "raw_coco_flower": int((res.get("raw_coco_counts") or {}).get("flower", 0)),
            "after_verify_flower": int((res.get("after_verify_counts") or {}).get("flower", 0)),
            "final_flower": int((res.get("after_finalize_counts") or {}).get("flower", 0)),
            "error": None,
            "stage_stats": res.get("stage_stats") or {},
            "flower_stage_counts": flower_stage_counts,
            "dominant_drop_stage": drop_stage,
            "dominant_drop_value": drop_val,
        }
        if item["gt_flower"] is not None:
            item["error"] = int(item["final_flower"] - int(item["gt_flower"]))
        items.append(item)

    # Top drops and examples
    top_drop_stages = sorted(stage_drop_acc.items(), key=lambda x: x[1], reverse=True)
    worst = sorted([x for x in items if x.get("error") is not None], key=lambda x: abs(int(x.get("error", 0))), reverse=True)

    report = {
        "summary": {
            "selected_total": len(items),
            "dataset_cases": len([x for x in items if x.get("source") == "dataset"]),
            "real_cases": len([x for x in items if x.get("source") == "real"]),
        },
        "top_drop_stages": [{"stage": k, "drop_sum": v} for k, v in top_drop_stages],
        "worst_cases": worst[:20],
        "items": items,
    }

    out_json = OUT_DIR / "flower_debug_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("selected_total", report["summary"]["selected_total"])
    print("dataset_cases", report["summary"]["dataset_cases"])
    print("real_cases", report["summary"]["real_cases"])
    print("top_drop_stages", report["top_drop_stages"][:5])
    print("report", out_json)
    print("snapshots", SNAP_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
