#!/usr/bin/env python3
from pathlib import Path
import random
import json
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import app as appmod
import numpy as np

ROOT = Path('S:/workspace/model_custom/dataset')
OUT = Path('S:/workspace/tmp_test_outputs/tree_sanity')
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path('S:/workspace/model_custom/weights/custo_all.pt')
SAMPLE_N = 30
IOU_THRESHOLD = 0.5


def read_label_boxes(lbl_path: Path):
    gts = []
    if not lbl_path.exists():
        return gts
    for line in lbl_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
        except Exception:
            continue
        gts.append({'class': cls, 'xywh': (x,y,w,h)})
    return gts


def xywh_norm_to_xyxy(xywh, img_w, img_h):
    x,y,w,h = xywh
    cx = x * img_w; cy = y * img_h
    bw = w * img_w; bh = h * img_h
    x1 = cx - bw/2; y1 = cy - bh/2; x2 = cx + bw/2; y2 = cy + bh/2
    return [x1,y1,x2,y2]


def iou(boxA, boxB):
    ax1,ay1,ax2,ay2 = boxA
    bx1,by1,bx2,by2 = boxB
    inter_x1 = max(ax1,bx1); inter_y1 = max(ay1,by1)
    inter_x2 = min(ax2,bx2); inter_y2 = min(ay2,by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    areaA = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    areaB = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


# collect candidate images that have GT class 3
candidates = []
for split in ('train','val'):
    lbl_dir = ROOT / split / 'labels'
    img_dir = ROOT / split / 'images'
    if not lbl_dir.exists():
        continue
    for lbl in sorted(lbl_dir.glob('*.txt')):
        gts = read_label_boxes(lbl)
        has_tree = any(gt['class'] == 3 for gt in gts)
        if has_tree:
            idx = lbl.stem
            # find image file
            found = None
            for ext in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'):
                p = img_dir / (idx + ext)
                if p.exists():
                    found = p
                    break
            if not found:
                # try any matching stem
                for p in img_dir.glob(idx + '.*'):
                    if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}:
                        found = p
                        break
            if found:
                candidates.append({'image': found, 'label': lbl, 'split': split})

if not candidates:
    print('No candidate images with class 3 found')
    raise SystemExit(1)

sample = random.sample(candidates, min(SAMPLE_N, len(candidates)))
print(f'Running sanity on {len(sample)} images')

# load model
model = YOLO(str(MODEL_PATH))
font = None
try:
    font = ImageFont.load_default()
except Exception:
    font = None

report = []

for item in sample:
    imgp = item['image']
    lblp = item['label']
    img = Image.open(imgp).convert('RGB')
    w,h = img.size
    # run custom model
    res = model.predict(source=str(imgp), conf=0.20, imgsz=640, device=0, verbose=False)[0]
    dets = []
    if getattr(res, 'boxes', None) is not None and len(res.boxes)>0:
        cls_list = res.boxes.cls.tolist()
        conf_list = res.boxes.conf.tolist()
        xyxy_list = [list(map(float, b.xyxy[0].cpu().numpy())) if hasattr(b, 'xyxy') else None for b in res.boxes]
        for cls, conf, xy in zip(cls_list, conf_list, xyxy_list):
            dets.append({'class': int(cls), 'conf': float(conf), 'xyxy': xy})

    # finalize via app pipeline (to match production)
    # convert to app expected format (class names used there), but finalize accepts class names; we'll map
    dets_named = []
    for d in dets:
        cname = str(model.model.names[int(d['class'])])
        dets_named.append({'class': cname, 'conf': d['conf'], 'xyxy': d['xyxy']})
    after_verify = appmod.verify_and_reduce_detections(dets_named, img_w=w, img_h=h, base_conf=0.35)
    finalize_ret = appmod.finalize_frame_detections_for_count(after_verify, img_w=w, img_h=h, min_conf=0.35, debug=True)
    if isinstance(finalize_ret, tuple):
        after_finalize = finalize_ret[0]
    else:
        after_finalize = finalize_ret

    # ground-truth boxes (class 3)
    gts = [gt for gt in read_label_boxes(lblp) if gt['class']==3]
    gt_xyxy = [xywh_norm_to_xyxy(gt['xywh'], w, h) for gt in gts]

    # match detections (from after_finalize) to GT
    det_xyxy = []
    det_conf = []
    for d in after_finalize:
        # d may have 'xyxy' normalized or absolute depending on pipeline; attempt to read
        xy = d.get('xyxy') or d.get('xywh') or []
        if not xy:
            continue
        if len(xy)==4:
            # detect whether normalized (0..1) or absolute >1
            if max(xy) <= 1.01:
                # convert norm xyxy to abs using image size
                ax = [xy[0]*w, xy[1]*h, xy[2]*w, xy[3]*h]
            else:
                ax = [float(x) for x in xy[:4]]
        else:
            continue
        det_xyxy.append(ax)
        det_conf.append(float(d.get('conf',0.0)))

    # compute matches
    matched_gt = [False]*len(gt_xyxy)
    matched_det = [False]*len(det_xyxy)
    for i, gt in enumerate(gt_xyxy):
        for j, det in enumerate(det_xyxy):
            if iou(gt, det) >= IOU_THRESHOLD:
                matched_gt[i] = True
                matched_det[j] = True
    misses = [gt for i,gt in enumerate(gt_xyxy) if not matched_gt[i]]
    false_pos = [det for j,det in enumerate(det_xyxy) if not matched_det[j]]

    # save overlay image
    draw = ImageDraw.Draw(img)
    # draw GT in green
    for gt in gt_xyxy:
        draw.rectangle(gt, outline='green', width=2)
    # draw dets in red
    for det,conf in zip(det_xyxy, det_conf):
        draw.rectangle(det, outline='red', width=2)
        txt = f"tree:{conf:.2f}"
        if font:
            draw.text((det[0]+2, det[1]+2), txt, fill='yellow', font=font)
        else:
            draw.text((det[0]+2, det[1]+2), txt, fill='yellow')
    out_path = OUT / imgp.name
    img.save(out_path)

    report.append({
        'image': str(imgp),
        'label': str(lblp),
        'num_gt': len(gt_xyxy),
        'num_det': len(det_xyxy),
        'misses': len(misses),
        'false_pos': len(false_pos),
        'out_overlay': str(out_path)
    })

# write report
OUT.joinpath('report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Sample check done. Report at', OUT / 'report.json')
