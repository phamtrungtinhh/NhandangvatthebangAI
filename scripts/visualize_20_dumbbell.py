#!/usr/bin/env python3
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

SAMPLE_JSON = Path('S:/workspace/tmp_test_outputs/dumbbell_infer_best.json')
OUT_DIR = Path('S:/workspace/tmp_test_outputs/visuals')
MODEL = Path('S:/workspace/model_custom/weights/custo_all.pt')
CONF_THR = 0.25

OUT_DIR.mkdir(parents=True, exist_ok=True)

payload = json.loads(SAMPLE_JSON.read_text(encoding='utf-8'))
images = [Path(x['image']) for x in payload.get('samples', [])][:20]
model = YOLO(str(MODEL))

try:
    font = ImageFont.truetype('arial.ttf', 16)
except Exception:
    font = ImageFont.load_default()

for i,img_path in enumerate(images, start=1):
    im = Image.open(img_path).convert('RGB')
    iw, ih = im.size
    draw = ImageDraw.Draw(im)
    # draw GT boxes
    label = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
    if label.exists():
        for line in label.read_text(encoding='utf-8', errors='ignore').splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
            except Exception:
                continue
            if cls != 0:
                continue
            bw = w * iw; bh = h * ih; cx = x * iw; cy = y * ih
            x1 = cx - bw/2; y1 = cy - bh/2; x2 = cx + bw/2; y2 = cy + bh/2
            draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
            draw.text((x1+3, max(0,y1+3)), 'GT:0', fill='green', font=font)
    # predict
    r = model.predict(source=str(img_path), imgsz=640, conf=CONF_THR, max_det=100, verbose=False)[0]
    if getattr(r, 'boxes', None) is not None and getattr(r.boxes, 'cls', None) is not None:
        cls_list = r.boxes.cls.tolist(); conf_list = r.boxes.conf.tolist(); xyxy_list = r.boxes.xyxy.tolist()
        for idx, c in enumerate(cls_list):
            if int(c) != 0:
                continue
            x1,y1,x2,y2 = xyxy_list[idx]
            conf = conf_list[idx]
            draw.rectangle([x1,y1,x2,y2], outline='red', width=3)
            draw.text((x1+3, max(0,y1+18)), f'P:0 {conf:.2f}', fill='red', font=font)
    outp = OUT_DIR / f'viz_{i:02d}_{img_path.name}'
    im.save(outp)
    print('wrote', outp)

print('Done: visuals in', OUT_DIR)
