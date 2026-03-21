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
OUT_JSON = OUT_DIR / "targeted_dumbbell_tree_report.json"

DEVICE = os.environ.get("FLOWER_RETEST_DEVICE", "cpu")
SEED = 20260316
BATCH_SIZE = 25
IOU_TP = float(os.environ.get("FLOWER_RETEST_IOU_TP", "0.50"))

ID_TO_NAME = {0: "dumbbell", 1: "flower", 2: "fruit", 3: "tree"}
PER_CLASS = ["flower", "tree", "fruit", "dumbbell"]
STAGE_ORDER = [
    "after_nms",
    "after_resolve_cross_class",
    "after_cross_class_suppress",
    "after_generic_fruit_overlap",
    "after_specific_fruit_conflict",
    "after_scene_level",
    "after_one_object_box",
    "after_dense_normalize",
    "after_suppress_flower_cross_class_confusions",
    "after_scene_rule_dedup",
    "after_flower_pruning",
    "after_scene_rules",
    "final",
]

TREE_TRACE_STEPS = [
    "start_after_verify",
    "enforce_one_object_one_box",
    "normalize_dense_flower_boxes",
    "suppress_flower_cross_class_confusions",
    "dedup_detections_by_class_nms_classwise",
    "refine_flower_boxes_with_visual_evidence",
]

CUSTOM_MODEL_PATH = os.environ.get("FLOWER_RETEST_CUSTOM_MODEL", "S:/workspace/model_custom/weights/custo_all.pt")
custom_model = YOLO(CUSTOM_MODEL_PATH)
coco_model = YOLO(str(app.COCO_MODEL_PATH))


def iou(a, b) -> float:
    return float(app._box_iou(a, b))


def xywhn_to_xyxy(x, y, w, h, iw, ih):
    x1 = (x - w / 2.0) * iw
    y1 = (y - h / 2.0) * ih
    x2 = (x + w / 2.0) * iw
    y2 = (y + h / 2.0) * ih
    return [float(x1), float(y1), float(x2), float(y2)]


def parse_labels(label_path: Path, iw: int, ih: int) -> dict[str, list[list[float]]]:
    out = defaultdict(list)
    for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = ln.strip().split()
        if len(p) < 5:
            continue
        try:
            cid = int(float(p[0]))
            x, y, w, h = map(float, p[1:5])
        except Exception:
            continue
        cname = ID_TO_NAME.get(cid)
        if cname is None:
            continue
        out[cname].append(xywhn_to_xyxy(x, y, w, h, iw, ih))
    return out


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def collect_candidates(target_class: str, min_gt: int = 1) -> list[dict]:
    out = []
    for split in ["train", "val"]:
        labels_dir = DATASET / split / "labels"
        images_dir = DATASET / split / "images"
        if not labels_dir.exists() or not images_dir.exists():
            continue
        for lbl in labels_dir.glob("*.txt"):
            img = find_image(images_dir, lbl.stem)
            if img is None:
                continue
            with Image.open(img).convert("RGB") as im:
                iw, ih = im.size
            gt = parse_labels(lbl, iw, ih)
            target_n = len(gt.get(target_class, []))
            if target_n < int(min_gt):
                continue
            out.append({
                "image": str(img),
                "label": str(lbl),
                "split": split,
                "gt": gt,
                "target_n": target_n,
            })
    return out


def run_pipeline(img_path: Path):
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

    merged = app.canonicalize_final_detections(custom_dets + coco_dets)
    raw_custom = app.canonicalize_final_detections(custom_dets)
    with Image.open(img_path).convert("RGB") as im:
        iw, ih = im.size
        image_np = None
        try:
            import numpy as np

            image_np = np.array(im)
        except Exception:
            pass

    scene_rule_debug = {}
    after_verify = app.verify_and_reduce_detections(
        merged,
        img_w=iw,
        img_h=ih,
        base_conf=0.35,
        scene_rule_debug=scene_rule_debug,
    )

    def _tree_count(ds):
        return int(
            sum(1 for d in (ds or []) if (app.normalize_class_name(d.get("class", "")) or "") == "tree")
        )

    # Trace tree drops by individual post-verify rules.
    tree_trace_counts = {}
    trace = app.canonicalize_final_detections(after_verify)
    tree_trace_counts["start_after_verify"] = _tree_count(trace)

    trace = app.enforce_one_object_one_box(trace, img_w=iw, img_h=ih)
    tree_trace_counts["enforce_one_object_one_box"] = _tree_count(trace)

    trace = app.normalize_dense_flower_boxes(trace, img_w=iw, img_h=ih)
    tree_trace_counts["normalize_dense_flower_boxes"] = _tree_count(trace)

    trace = app.suppress_flower_cross_class_confusions(trace, img_w=iw, img_h=ih, base_conf=0.35)
    tree_trace_counts["suppress_flower_cross_class_confusions"] = _tree_count(trace)

    trace = app.dedup_detections_by_class_nms_classwise(trace, default_iou=0.70)
    tree_trace_counts["dedup_detections_by_class_nms_classwise"] = _tree_count(trace)

    if image_np is not None:
        trace = app.refine_flower_boxes_with_visual_evidence(image_np, trace, img_w=iw, img_h=ih)
    tree_trace_counts["refine_flower_boxes_with_visual_evidence"] = _tree_count(trace)

    tree_trace_drop_counts = {}
    prev = None
    for s in TREE_TRACE_STEPS:
        cur = int(tree_trace_counts.get(s, 0))
        if prev is not None and prev > cur:
            tree_trace_drop_counts[s] = int(prev - cur)
        prev = cur

    ret = app.finalize_frame_detections_for_count(after_verify, img_w=iw, img_h=ih, min_conf=0.35, debug=True)
    if isinstance(ret, tuple) and len(ret) == 3:
        final_dets, stage_stats, stage_dets = ret
    else:
        final_dets, stage_stats, stage_dets = ret, {}, {}
    return {
        "final_dets": app.canonicalize_final_detections(final_dets or []),
        "stage_stats": stage_stats or {},
        "stage_dets": stage_dets or {},
        "scene_rule_debug": scene_rule_debug,
        "raw_custom_counts": count_classes(raw_custom),
        "raw_custom_dumbbell": int(sum(1 for d in raw_custom if (app.normalize_class_name(d.get("class", "")) or "") == "dumbbell")),
        "tree_trace_counts": tree_trace_counts,
        "tree_trace_drop_counts": tree_trace_drop_counts,
    }


def count_classes(dets: list[dict]) -> dict[str, int]:
    c = Counter()
    for d in dets:
        cls = app.normalize_class_name(d.get("class", "")) or "unknown"
        c[cls] += 1
    return dict(c)


def boxes_for_class(dets: list[dict], class_name: str) -> list[list[float]]:
    out = []
    for d in dets:
        cls = app.normalize_class_name(d.get("class", "")) or "unknown"
        if cls != class_name:
            continue
        xy = d.get("xyxy")
        if isinstance(xy, (list, tuple)) and len(xy) >= 4:
            out.append([float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])])
    return out


def match_tp_fp_fn(gt_boxes: list[list[float]], pred_boxes: list[list[float]], iou_thr: float = IOU_TP):
    matched_gt = set()
    matched_pred = set()
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            pairs.append((iou(pb, gb), pi, gi))
    pairs.sort(key=lambda x: x[0], reverse=True)
    tp = 0
    for ov, pi, gi in pairs:
        if ov < iou_thr:
            break
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        tp += 1
    fp = max(0, len(pred_boxes) - tp)
    fn = max(0, len(gt_boxes) - tp)
    return tp, fp, fn


def class_stage_counts(stage_stats: dict, target_class: str) -> dict[str, int]:
    cc = stage_stats.get("class_counts", {}) if isinstance(stage_stats, dict) else {}
    out = {}
    for s in STAGE_ORDER:
        out[s] = int((cc.get(s, {}) or {}).get(target_class, 0)) if isinstance(cc.get(s, {}), dict) else 0
    return out


def dominant_drop(stage_counts: dict[str, int]) -> tuple[str | None, int]:
    prev = None
    best_s = None
    best_d = 0
    for s in STAGE_ORDER:
        cur = int(stage_counts.get(s, 0))
        if prev is not None:
            d = prev - cur
            if d > best_d:
                best_d = d
                best_s = s
        prev = cur
    return best_s, best_d


def evaluate_batch(target_class: str, candidates: list[dict], batch_size: int, note_tree_flower: bool = False):
    random.seed(SEED)
    # prefer images with larger target count to stress conditions
    ranked = sorted(candidates, key=lambda x: int(x.get("target_n", 0)), reverse=True)
    top = ranked[: max(batch_size * 2, batch_size)]
    random.shuffle(top)
    selected = top[:batch_size]

    tot_tp = tot_fp = tot_fn = 0
    per_class_totals = Counter()
    stage_drop_acc = Counter()
    fallback_count = 0
    items = []
    tree_suppression_notes = []
    scene_rule_drop_acc = Counter()
    tree_trace_drop_acc = Counter()
    raw_custom_dumbbell_total = 0
    raw_custom_dumbbell_nonzero_images = 0

    for it in selected:
        img = Path(it["image"])
        gt_map = it.get("gt", {})
        gt_boxes = gt_map.get(target_class, [])

        run = run_pipeline(img)
        final_dets = run.get("final_dets", [])
        stage_stats = run.get("stage_stats", {})
        scene_rule_debug = run.get("scene_rule_debug", {})
        tree_trace_counts = run.get("tree_trace_counts", {})
        tree_trace_drop_counts = run.get("tree_trace_drop_counts", {})
        raw_custom_counts = run.get("raw_custom_counts", {})
        raw_custom_dumbbell = int(run.get("raw_custom_dumbbell", 0) or 0)

        raw_custom_dumbbell_total += raw_custom_dumbbell
        if raw_custom_dumbbell > 0:
            raw_custom_dumbbell_nonzero_images += 1

        pred_boxes = boxes_for_class(final_dets, target_class)
        iou_match = 0.40 if target_class == "dumbbell" else IOU_TP
        tp, fp, fn = match_tp_fp_fn(gt_boxes, pred_boxes, iou_thr=iou_match)

        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        fallback_count += 1 if bool(stage_stats.get("fallback_used")) else 0

        cc = count_classes(final_dets)
        for c in PER_CLASS:
            per_class_totals[c] += int(cc.get(c, 0))

        st_cnt = class_stage_counts(stage_stats, target_class)
        ds, dv = dominant_drop(st_cnt)
        if ds and dv > 0:
            stage_drop_acc[ds] += dv

        if note_tree_flower:
            for k, v in (scene_rule_debug or {}).items():
                if str(k).startswith("drop_tree_"):
                    scene_rule_drop_acc[str(k)] += int(v)
            for k, v in ((stage_stats.get("tree_drop_reasons_after_scene_rules", {}) if isinstance(stage_stats, dict) else {}) or {}).items():
                scene_rule_drop_acc[f"after_scene_rules::{k}"] += int(v)
            for k, v in (tree_trace_drop_counts or {}).items():
                tree_trace_drop_acc[str(k)] += int(v)
            tree_before = int(st_cnt.get("after_flower_pruning", 0))
            tree_after = int(st_cnt.get("after_scene_rules", 0))
            flower_after = int((stage_stats.get("class_counts", {}).get("after_scene_rules", {}) or {}).get("flower", 0))
            if tree_after < tree_before:
                tree_suppression_notes.append(
                    {
                        "image": str(img),
                        "tree_drop": int(tree_before - tree_after),
                        "flower_after_scene_rules": flower_after,
                        "stage_drop": {"stage": ds, "drop": int(dv)},
                        "scene_rule_debug": scene_rule_debug,
                        "after_scene_rule_debug": stage_stats.get("tree_drop_reasons_after_scene_rules", {}),
                        "tree_trace_drop_counts": tree_trace_drop_counts,
                    }
                )

        items.append(
            {
                "image": str(img),
                "split": it.get("split"),
                "gt_target": len(gt_boxes),
                "pred_target": len(pred_boxes),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "per_class_counts": {c: int(cc.get(c, 0)) for c in PER_CLASS},
                "stage_counts_target": st_cnt,
                "dominant_drop_stage": ds,
                "dominant_drop_value": int(dv),
                "fallback_used": bool(stage_stats.get("fallback_used")),
                "scene_rule_debug": scene_rule_debug,
                "tree_trace_counts": tree_trace_counts,
                "tree_trace_drop_counts": tree_trace_drop_counts,
                "raw_custom_counts": raw_custom_counts,
                "raw_custom_dumbbell": raw_custom_dumbbell,
            }
        )

    precision = float(tot_tp) / float(max(1, tot_tp + tot_fp))
    recall = float(tot_tp) / float(max(1, tot_tp + tot_fn))
    top_drop = [{"stage": k, "drop_sum": v} for k, v in stage_drop_acc.most_common()]

    # worst by target error + fn weight
    worst = sorted(items, key=lambda x: (x.get("fn", 0) + x.get("fp", 0), x.get("fn", 0)), reverse=True)[:3]

    return {
        "target_class": target_class,
        "batch_size": len(items),
        "tp": int(tot_tp),
        "fp": int(tot_fp),
        "fn": int(tot_fn),
        "precision": precision,
        "recall": recall,
        "per_class_counts_total": {c: int(per_class_totals.get(c, 0)) for c in PER_CLASS},
        "top_drop_stages": top_drop,
        "fallback_count": int(fallback_count),
        "worst_3_cases": [
            {
                "image": x.get("image"),
                "tp": x.get("tp"),
                "fp": x.get("fp"),
                "fn": x.get("fn"),
                "dominant_drop_stage": x.get("dominant_drop_stage"),
                "dominant_drop_value": x.get("dominant_drop_value"),
            }
            for x in worst
        ],
        "tree_suppression_notes": tree_suppression_notes if note_tree_flower else [],
        "tree_scene_rule_drop_counts": dict(scene_rule_drop_acc) if note_tree_flower else {},
        "tree_trace_step_drop_counts": dict(tree_trace_drop_acc) if note_tree_flower else {},
        "raw_custom_dumbbell_total": int(raw_custom_dumbbell_total),
        "raw_custom_dumbbell_nonzero_images": int(raw_custom_dumbbell_nonzero_images),
        "items": items,
    }


def main():
    dumbbell_candidates = collect_candidates("dumbbell", min_gt=1)
    tree_candidates = collect_candidates("tree", min_gt=3)

    dumbbell_report = evaluate_batch("dumbbell", dumbbell_candidates, batch_size=BATCH_SIZE, note_tree_flower=False)
    tree_report = evaluate_batch("tree", tree_candidates, batch_size=BATCH_SIZE, note_tree_flower=True)

    out = {
        "config": {
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "iou_tp": IOU_TP,
            "seed": SEED,
            "custom_model_path": CUSTOM_MODEL_PATH,
        },
        "dumbbell": dumbbell_report,
        "tree": tree_report,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("wrote", OUT_JSON)
    print(json.dumps(
        {
            "dumbbell": {
                "batch_size": dumbbell_report["batch_size"],
                "tp": dumbbell_report["tp"],
                "fp": dumbbell_report["fp"],
                "fn": dumbbell_report["fn"],
                "precision": round(dumbbell_report["precision"], 4),
                "recall": round(dumbbell_report["recall"], 4),
                "top_drop_stages": dumbbell_report["top_drop_stages"][:3],
            },
            "tree": {
                "batch_size": tree_report["batch_size"],
                "tp": tree_report["tp"],
                "fp": tree_report["fp"],
                "fn": tree_report["fn"],
                "precision": round(tree_report["precision"], 4),
                "recall": round(tree_report["recall"], 4),
                "top_drop_stages": tree_report["top_drop_stages"][:3],
                "tree_scene_rule_drop_counts": tree_report.get("tree_scene_rule_drop_counts", {}),
                "tree_trace_step_drop_counts": tree_report.get("tree_trace_step_drop_counts", {}),
                "tree_suppression_notes_count": len(tree_report.get("tree_suppression_notes", [])),
            },
            "dumbbell_raw_custom": {
                "raw_custom_dumbbell_total": dumbbell_report.get("raw_custom_dumbbell_total"),
                "raw_custom_dumbbell_nonzero_images": dumbbell_report.get("raw_custom_dumbbell_nonzero_images"),
            },
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
