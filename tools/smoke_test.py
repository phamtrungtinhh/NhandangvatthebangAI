import json
import sys
from pathlib import Path

import app


def _read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def _safe_stage_stats(raw):
    if isinstance(raw, dict):
        return raw.get("stage_stats", {}) or {}
    return {}


def run_smoke(paths):
    coco_model, custom_model = app.load_default_detection_models(
        coco_cache_token=app.get_model_cache_token(app.COCO_MODEL_PATH),
        custom_cache_token=app.get_model_cache_token(app.CUSTOM_MODEL_PATH),
    )

    results = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            results.append({"file": str(p), "error": "not_found"})
            continue

        data = _read_bytes(str(p))

        ann_coco, counts_coco, raw_coco, raw_img_coco = app.process_uploaded(
            coco_model,
            data,
            True,
            conf=0.25,
            imgsz=640,
            use_fp16=False,
            max_frames=1,
            sample_rate=1,
            batch_size=1,
            nms_iou=0.5,
            max_det=130,
        )

        ann_custom, counts_custom, raw_custom, raw_img_custom = app.process_uploaded(
            custom_model,
            data,
            True,
            conf=0.25,
            imgsz=640,
            use_fp16=False,
            max_frames=1,
            sample_rate=1,
            batch_size=1,
            nms_iou=0.5,
            max_det=130,
        )

        raw_img = raw_img_custom if raw_img_custom is not None else raw_img_coco
        img_h, img_w = (raw_img.shape[0], raw_img.shape[1]) if raw_img is not None else (0, 0)

        merged = []
        if isinstance(raw_coco, dict):
            merged.extend(raw_coco.get("detections", []) or [])
        if isinstance(raw_custom, dict):
            merged.extend(raw_custom.get("detections", []) or [])

        merged = app.canonicalize_final_detections(merged)
        before_counts = app.build_counts_from_detections(merged)
        after_rescue = app.rescue_fruit_from_coco_when_flower_only(
            merged,
            raw_coco.get("detections", []) if isinstance(raw_coco, dict) else [],
            img_w=img_w,
            img_h=img_h,
            base_conf=0.25,
        )
        after_counts = app.build_counts_from_detections(after_rescue)

        results.append(
            {
                "file": str(p),
                "custom_stage_stats": _safe_stage_stats(raw_custom),
                "coco_stage_stats": _safe_stage_stats(raw_coco),
                "merged_counts_before_rescue": before_counts,
                "merged_counts_after_rescue": after_counts,
            }
        )

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_paths = sys.argv[1:]
    else:
        img_paths = [
            r"S:\hoa test\anh-nen-hoa-tulip-4k_014120128.jpg",
            r"S:\hoa test\hinh-anh-hoa-dep-nhat_110818349.jpg",
            r"S:\hoa test\chuoi2.png",
            r"S:\hoa test\nguoi va hoa.jpg",
        ]

    out = run_smoke(img_paths)
    print(json.dumps(out, ensure_ascii=False, indent=2))
