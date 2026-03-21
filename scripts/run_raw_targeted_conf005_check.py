#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from pathlib import Path
import importlib.util

from ultralytics import YOLO

DATASET = Path("S:/workspace/model_custom/dataset")
OUT_DIR = Path("S:/workspace/tmp_test_outputs/flower_rule_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "targeted_raw_detect_conf005_report.json"

DEVICE = os.environ.get("FLOWER_RETEST_DEVICE", "0")
SEED = 20260316
BATCH_SIZE = 25
IOU_TP = float(os.environ.get("FLOWER_RETEST_IOU_TP", "0.50"))
RAW_CONF = float(os.environ.get("FLOWER_RAW_CONF", "0.05"))
CUSTOM_MODEL_PATH = os.environ.get(
    "FLOWER_RETEST_CUSTOM_MODEL",
    "S:/workspace/model_custom/_focused_runs/dumbbell_tree_oversample_x3_e16/weights/best.pt",
)

spec = importlib.util.spec_from_file_location("targeted", "S:/workspace/scripts/run_targeted_dumbbell_tree_checks.py")
targeted = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(targeted)

custom_model = YOLO(CUSTOM_MODEL_PATH)


def canonical_name(name: str) -> str:
    return str(targeted.app.normalize_class_name(name) or "").strip()


def raw_predict(img_path: Path) -> list[dict]:
    res = custom_model.predict(source=str(img_path), conf=RAW_CONF, imgsz=512, device=DEVICE, verbose=False)
    r = res[0]
    dets = []
    if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
        return dets

    cls_list = r.boxes.cls.tolist()
    conf_list = r.boxes.conf.tolist()
    xyxy_list = [list(map(float, b.xyxy[0].cpu().numpy())) if hasattr(b, "xyxy") else None for b in r.boxes]
    for idx, (cls_id, conf) in enumerate(zip(cls_list, conf_list)):
        cname = str(custom_model.model.names[int(cls_id)])
        xy = xyxy_list[idx] if idx < len(xyxy_list) else None
        dets.append({"class": canonical_name(cname), "conf": float(conf), "xyxy": xy})
    return dets


def boxes_for_class(dets: list[dict], target_class: str) -> list[list[float]]:
    out = []
    for d in dets:
        if str(d.get("class", "")) != target_class:
            continue
        xy = d.get("xyxy")
        if isinstance(xy, (list, tuple)) and len(xy) >= 4:
            out.append([float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])])
    return out


def evaluate_raw(target_class: str, min_gt: int) -> dict:
    candidates = targeted.collect_candidates(target_class, min_gt=min_gt)
    random.seed(SEED)
    ranked = sorted(candidates, key=lambda x: int(x.get("target_n", 0)), reverse=True)
    top = ranked[: max(BATCH_SIZE * 2, BATCH_SIZE)]
    random.shuffle(top)
    selected = top[:BATCH_SIZE]

    tot_tp = tot_fp = tot_fn = 0
    items = []
    nonzero_pred_images = 0

    for it in selected:
        img = Path(it["image"])
        gt_boxes = (it.get("gt") or {}).get(target_class, [])
        dets = raw_predict(img)
        pred_boxes = boxes_for_class(dets, target_class)
        iou_match = 0.40 if target_class == "dumbbell" else IOU_TP
        tp, fp, fn = targeted.match_tp_fp_fn(gt_boxes, pred_boxes, iou_thr=iou_match)

        tot_tp += int(tp)
        tot_fp += int(fp)
        tot_fn += int(fn)
        if len(pred_boxes) > 0:
            nonzero_pred_images += 1

        items.append(
            {
                "image": str(img),
                "split": it.get("split"),
                "gt_target": int(len(gt_boxes)),
                "pred_target_raw": int(len(pred_boxes)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )

    precision = float(tot_tp) / float(max(1, tot_tp + tot_fp))
    recall = float(tot_tp) / float(max(1, tot_tp + tot_fn))

    return {
        "target_class": target_class,
        "batch_size": len(items),
        "raw_conf": RAW_CONF,
        "tp": int(tot_tp),
        "fp": int(tot_fp),
        "fn": int(tot_fn),
        "precision": precision,
        "recall": recall,
        "nonzero_pred_images": int(nonzero_pred_images),
        "items": items,
    }


def main() -> int:
    dumbbell = evaluate_raw("dumbbell", min_gt=1)
    tree = evaluate_raw("tree", min_gt=3)

    out = {
        "config": {
            "device": DEVICE,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "iou_tp": IOU_TP,
            "raw_conf": RAW_CONF,
            "custom_model_path": CUSTOM_MODEL_PATH,
        },
        "dumbbell_raw": dumbbell,
        "tree_raw": tree,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("wrote", OUT_JSON)
    print(
        json.dumps(
            {
                "dumbbell_raw": {
                    "tp": dumbbell["tp"],
                    "fp": dumbbell["fp"],
                    "fn": dumbbell["fn"],
                    "recall": round(dumbbell["recall"], 6),
                    "nonzero_pred_images": dumbbell["nonzero_pred_images"],
                },
                "tree_raw": {
                    "tp": tree["tp"],
                    "fp": tree["fp"],
                    "fn": tree["fn"],
                    "recall": round(tree["recall"], 6),
                    "nonzero_pred_images": tree["nonzero_pred_images"],
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
