#!/usr/bin/env python3
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from ultralytics import YOLO

MODEL = Path('S:/workspace/model_custom/weights/custo_all.pt')
VAL_DIR = Path('S:/workspace/model_custom/dataset/val/images')
OUT = Path('S:/workspace/model_custom/weights/val_bias_check.json')
SAMPLE_N = 100
IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.gif','.webp','.tif','.tiff'}

def gather_images(d):
    if not d.exists():
        return []
    return [p for p in d.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS]


def main():
    imgs = gather_images(VAL_DIR)
    n = min(SAMPLE_N, len(imgs))
    if n == 0:
        print('No images found in', VAL_DIR)
        return
    sample = random.sample(imgs, n)

    model = YOLO(str(MODEL))

    counts = Counter()
    conf_sums = defaultdict(float)
    total_dets = 0

    items = []
    for p in sample:
        res = model.predict(source=str(p), conf=0.35, imgsz=512, device=0, verbose=False)
        r = res[0]
        dets = []
        if getattr(r, 'boxes', None) is not None and len(r.boxes) > 0:
            cls_list = r.boxes.cls.tolist()
            conf_list = r.boxes.conf.tolist()
            for cls, conf in zip(cls_list, conf_list):
                cls_i = int(cls)
                counts[cls_i] += 1
                conf_sums[cls_i] += float(conf)
                total_dets += 1
                dets.append({'class_id': cls_i, 'conf': float(conf)})
        items.append({'image': str(p), 'detections': dets, 'num_detections': len(dets)})

    # compute stats
    stat = {'total_images': n, 'total_detections': total_dets, 'by_class': {}, 'percentage': {}, 'avg_confidence': {}}
    for k in sorted(list(set(list(counts.keys()) + [0,1,2,3]))):
        c = counts.get(k, 0)
        stat['by_class'][str(k)] = c
        stat['percentage'][str(k)] = (c / total_dets * 100) if total_dets > 0 else 0.0
        stat['avg_confidence'][str(k)] = (conf_sums.get(k,0.0) / c) if c>0 else 0.0

    # warning
    warning = None
    for k,v in stat['by_class'].items():
        if total_dets>0 and (v / total_dets) > 0.7:
            warning = 'WARNING: possible model prediction bias'
            break

    out = {'model': str(MODEL), 'sample_images': n, 'total_detections': total_dets, 'by_class': stat['by_class'], 'percentage': stat['percentage'], 'avg_confidence': stat['avg_confidence'], 'warning': warning, 'items': items}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')

    # print table
    print('Class | Detections | Percentage | Avg Confidence')
    print('-----------------------------------------------')
    class_names = {0:'dumbbell',1:'flower',2:'fruit',3:'tree'}
    for k in sorted(stat['by_class'].keys(), key=lambda x:int(x)):
        name = class_names.get(int(k), str(k))
        det = stat['by_class'][k]
        pct = stat['percentage'][k]
        avgc = stat['avg_confidence'][k]
        print(f"{k} ({name}) | {det} | {pct:.2f}% | {avgc:.3f}")

    print('\nTotal images:', n)
    print('Total detections:', total_dets)
    if warning:
        print('\n'+warning)
    print('\nWritten log to', OUT)

if __name__ == '__main__':
    main()
