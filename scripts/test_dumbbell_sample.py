#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import json
import random

from ultralytics import YOLO

ROOT = Path('S:/workspace/model_custom/dataset')
WEIGHTS = Path('S:/workspace/model_custom/weights/custo_all.pt')
OUT = Path('S:/workspace/tmp_test_outputs/dumbbell_infer.json')

# find candidate images that have dumbbell labels
candidates = []
for split in ['train','val']:
    labels_dir = ROOT/split/'labels'
    images_dir = ROOT/split/'images'
    if not labels_dir.exists():
        continue
    for lbl in labels_dir.glob('*.txt'):
        txt = lbl.read_text(encoding='utf-8',errors='ignore')
        has = False
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln: continue
            parts = ln.split()
            if len(parts) < 5: continue
            try:
                cls = int(float(parts[0]))
            except:
                continue
            if cls == 0:
                # find image
                stem = lbl.stem
                img = None
                for ext in ['.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff']:
                    cand = images_dir / (stem+ext)
                    if cand.exists():
                        img = cand
                        break
                if img is None:
                    # try any file with same stem
                    for p in images_dir.glob(stem+'*'):
                        if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'):
                            img = p
                            break
                if img is not None:
                    candidates.append(str(img))
                break

if not candidates:
    print('No dumbbell images found in dataset')
    raise SystemExit(1)

random.seed(42)
sample = random.sample(candidates, min(20, len(candidates)))
print('Testing', len(sample), 'images')
model = YOLO(str(WEIGHTS))
reslist = []
summary = defaultdict(list)

for img in sample:
    try:
        results = model.predict(source=img, imgsz=640, conf=0.25, max_det=100)
    except Exception:
        # fallback
        results = model(img)
    r = results[0]
    detections = []
    # ultralytics v8: r.boxes.cls, r.boxes.conf
    boxes = getattr(r, 'boxes', None)
    if boxes is not None:
        cls_arr = getattr(boxes, 'cls', None)
        conf_arr = getattr(boxes, 'conf', None)
        xyxy = getattr(boxes, 'xyxy', None)
        if cls_arr is not None:
            for i, c in enumerate(cls_arr.tolist() if hasattr(cls_arr, 'tolist') else list(cls_arr)):
                conf = conf_arr[i].item() if hasattr(conf_arr[i], 'item') else float(conf_arr[i])
                clsid = int(c)
                detections.append({'class': clsid, 'conf': float(conf)})
    else:
        # try r.boxes.data
        try:
            data = r.boxes.data.tolist()
            for row in data:
                x1,y1,x2,y2,conf,clsid = row
                detections.append({'class': int(clsid), 'conf': float(conf)})
        except Exception:
            pass
    reslist.append({'image': img, 'detections': detections})
    for d in detections:
        summary[d['class']].append(d['conf'])

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({'samples': reslist, 'summary': {str(k): {'count': len(v), 'avg_conf': sum(v)/len(v) if v else 0} for k,v in summary.items()}}, indent=2), encoding='utf-8')
print('Wrote', OUT)
print('Summary:')
for k,v in summary.items():
    print('class',k,'count',len(v),'avg_conf',sum(v)/len(v))
