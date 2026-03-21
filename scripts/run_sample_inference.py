#!/usr/bin/env python3
import json
import random
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

MODEL = Path('S:/workspace/model_custom/weights/custo_all.pt')
DATASET_VAL = Path('S:/workspace/model_custom/dataset_flower_focused/val/images')
OUT_DIR = Path('S:/workspace/tmp_test_outputs/sample_inference')
N = 12
CONF = 0.35
IMGSZ = 640
DEVICE = '0'


def gather_images(d):
    if not d.exists():
        return []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    return [p for p in sorted(d.rglob('*')) if p.suffix.lower() in exts]


def main():
    imgs = gather_images(DATASET_VAL)
    if not imgs:
        print('No images found in', DATASET_VAL)
        return
    sample = random.sample(imgs, min(N, len(imgs)))
    model = YOLO(str(MODEL))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_summary = []
    for p in sample:
        res = model.predict(source=str(p), conf=CONF, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
        dets = []
        if getattr(res, 'boxes', None) is not None and len(res.boxes) > 0:
            cls_list = res.boxes.cls.tolist()
            conf_list = res.boxes.conf.tolist()
            xyxy = res.boxes.xyxy.tolist()
            for cls, conf, box in zip(cls_list, conf_list, xyxy):
                dets.append({'class_id': int(cls), 'conf': float(conf), 'xyxy': [float(x) for x in box]})
        # plot and save annotated image
        try:
            img_arr = res.plot()
            img = Image.fromarray(img_arr)
            out_path = OUT_DIR / (p.stem + '_annotated.png')
            img.save(out_path)
        except Exception:
            out_path = None
        results_summary.append({'image': str(p), 'annotated': str(out_path) if out_path else None, 'detections': dets})

    out_json = Path('S:/workspace/model_custom/weights/sample_inference_20260312.json')
    out_json.write_text(json.dumps(results_summary, indent=2), encoding='utf-8')
    print('Wrote', out_json)
    print('Annotated images saved to', OUT_DIR)


if __name__ == '__main__':
    main()
