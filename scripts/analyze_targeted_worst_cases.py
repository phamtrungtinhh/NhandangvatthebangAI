#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import importlib.util
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPORT_PATH = Path("S:/workspace/tmp_test_outputs/flower_rule_debug/targeted_dumbbell_tree_report.json")
OUT_DIR = Path("S:/workspace/tmp_test_outputs/flower_rule_debug/worst_cases_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_targeted_module(model_path: str, device: str):
    os.environ["FLOWER_RETEST_CUSTOM_MODEL"] = model_path
    os.environ["FLOWER_RETEST_DEVICE"] = device
    spec = importlib.util.spec_from_file_location(
        "run_targeted_dumbbell_tree_checks", "S:/workspace/scripts/run_targeted_dumbbell_tree_checks.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _image_quality(arr: np.ndarray) -> dict[str, float]:
    gray = arr.mean(axis=2).astype(np.float32)
    # Simple Laplacian variance proxy for blur.
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "laplacian_var": float(lap.var()),
    }


def _area_stats(boxes: list[list[float]], iw: int, ih: int) -> dict[str, float]:
    img_area = float(max(1, iw * ih))
    ratios = []
    for b in boxes:
        x1, y1, x2, y2 = b
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        ratios.append((w * h) / img_area)
    if not ratios:
        return {"count": 0, "min": 0.0, "median": 0.0, "max": 0.0, "small_lt_0.001": 0}
    arr = np.array(ratios, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "small_lt_0.001": int((arr < 0.001).sum()),
    }


def _match_pairs(rt: Any, gt_boxes: list[list[float]], pred_boxes: list[list[float]], iou_thr: float) -> dict[str, Any]:
    matched_gt = set()
    matched_pred = set()
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            pairs.append((float(rt.iou(pb, gb)), pi, gi))
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched = []
    for ov, pi, gi in pairs:
        if ov < iou_thr:
            break
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matched.append({"pred_idx": int(pi), "gt_idx": int(gi), "iou": float(ov)})

    return {
        "matched": matched,
        "unmatched_gt": [i for i in range(len(gt_boxes)) if i not in matched_gt],
        "unmatched_pred": [i for i in range(len(pred_boxes)) if i not in matched_pred],
    }


def _draw_overlay(
    image_path: Path,
    gt_boxes: list[list[float]],
    pred_dets: list[dict[str, Any]],
    target_class: str,
    out_path: Path,
):
    with Image.open(image_path).convert("RGB") as im:
        draw = ImageDraw.Draw(im)

        for b in gt_boxes:
            draw.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 60, 60), width=2)
            draw.text((b[0], max(0.0, b[1] - 12.0)), "GT", fill=(255, 60, 60))

        for d in pred_dets:
            cls = str(d.get("class", ""))
            if cls != target_class:
                continue
            xy = d.get("xyxy")
            if not (isinstance(xy, (list, tuple)) and len(xy) >= 4):
                continue
            conf = float(d.get("conf", 0.0))
            draw.rectangle([xy[0], xy[1], xy[2], xy[3]], outline=(60, 255, 60), width=2)
            draw.text((xy[0], max(0.0, float(xy[1]) - 12.0)), f"P {conf:.2f}", fill=(60, 255, 60))

        im.save(out_path)


def _label_path_from_image(image_path: Path) -> Path:
    p = str(image_path)
    p = p.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
    return Path(p).with_suffix(".txt")


def main() -> int:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    model_path = str(report.get("config", {}).get("custom_model_path", "")).strip()
    device = str(report.get("config", {}).get("device", "0")).strip() or "0"

    if not model_path:
        raise RuntimeError("Report has no custom_model_path")

    rt = _load_targeted_module(model_path=model_path, device=device)

    out: dict[str, Any] = {
        "report_path": str(REPORT_PATH),
        "model_path": model_path,
        "device": device,
        "classes": {},
    }

    for cls_name in ["dumbbell", "tree"]:
        cls_report = report.get(cls_name, {})
        worst_cases = cls_report.get("worst_3_cases", []) or []
        cls_out_cases = []

        for idx, wc in enumerate(worst_cases, start=1):
            img_path = Path(str(wc.get("image", "")))
            if not img_path.exists():
                continue

            with Image.open(img_path).convert("RGB") as im:
                iw, ih = im.size
                arr = np.array(im)

            label_path = _label_path_from_image(img_path)
            gt_map = rt.parse_labels(label_path, iw, ih) if label_path.exists() else {}
            gt_boxes = gt_map.get(cls_name, [])

            run = rt.run_pipeline(img_path)
            final_dets = run.get("final_dets", []) or []
            pred_boxes = rt.boxes_for_class(final_dets, cls_name)
            iou_thr = 0.40 if cls_name == "dumbbell" else float(report.get("config", {}).get("iou_tp", 0.5))
            tp, fp, fn = rt.match_tp_fp_fn(gt_boxes, pred_boxes, iou_thr=iou_thr)

            stage_counts = rt.class_stage_counts(run.get("stage_stats", {}) or {}, cls_name)
            drop_stage, drop_value = rt.dominant_drop(stage_counts)
            matched = _match_pairs(rt, gt_boxes, pred_boxes, iou_thr=iou_thr)

            overlay_name = f"{cls_name}_{idx}_{img_path.stem}_gt_pred.jpg"
            overlay_path = OUT_DIR / overlay_name
            _draw_overlay(img_path, gt_boxes, final_dets, cls_name, overlay_path)

            area_stats = _area_stats(gt_boxes, iw, ih)
            quality = _image_quality(arr)

            likely_causes = []
            if len(gt_boxes) == 0:
                likely_causes.append("label_missing_or_parse_error")
            if int(stage_counts.get("after_nms", 0)) == 0:
                likely_causes.append("no_initial_detection")
            if area_stats["count"] > 0 and area_stats["median"] < 0.001:
                likely_causes.append("tiny_objects_dominant")
            if quality["laplacian_var"] < 80.0:
                likely_causes.append("possible_blur")
            if quality["brightness_mean"] < 45.0:
                likely_causes.append("very_dark_image")
            if drop_stage is not None and int(drop_value) > 0:
                likely_causes.append("post_rule_suppression")

            cls_out_cases.append(
                {
                    "image": str(img_path),
                    "overlay": str(overlay_path),
                    "label": str(label_path),
                    "gt_count": int(len(gt_boxes)),
                    "pred_count": int(len(pred_boxes)),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "iou_thr": float(iou_thr),
                    "stage_counts_target": stage_counts,
                    "dominant_drop_stage": drop_stage,
                    "dominant_drop_value": int(drop_value or 0),
                    "area_stats": area_stats,
                    "quality": quality,
                    "matched_info": matched,
                    "scene_rule_debug": run.get("scene_rule_debug", {}),
                    "tree_trace_drop_counts": run.get("tree_trace_drop_counts", {}),
                    "raw_custom_counts": run.get("raw_custom_counts", {}),
                    "likely_causes": likely_causes,
                }
            )

        cause_counter: dict[str, int] = {}
        for c in cls_out_cases:
            for cause in c.get("likely_causes", []):
                cause_counter[cause] = int(cause_counter.get(cause, 0)) + 1

        out["classes"][cls_name] = {
            "summary": {
                "cases": len(cls_out_cases),
                "tp_sum": int(sum(x.get("tp", 0) for x in cls_out_cases)),
                "fp_sum": int(sum(x.get("fp", 0) for x in cls_out_cases)),
                "fn_sum": int(sum(x.get("fn", 0) for x in cls_out_cases)),
                "cause_counter": cause_counter,
            },
            "cases": cls_out_cases,
        }

    out_json = OUT_DIR / "worst_cases_analysis.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Worst Cases Analysis",
        "",
        f"- Report: {REPORT_PATH}",
        f"- Model: {model_path}",
        "",
    ]
    for cls_name in ["dumbbell", "tree"]:
        cs = out["classes"].get(cls_name, {}).get("summary", {})
        lines.append(f"## {cls_name}")
        lines.append(f"- cases: {cs.get('cases', 0)}")
        lines.append(f"- tp/fp/fn: {cs.get('tp_sum', 0)}/{cs.get('fp_sum', 0)}/{cs.get('fn_sum', 0)}")
        lines.append(f"- cause_counter: {cs.get('cause_counter', {})}")
        lines.append("")

    out_md = OUT_DIR / "worst_cases_analysis.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("wrote", out_json)
    print("wrote", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
