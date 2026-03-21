import json
from ultralytics import YOLO
from pathlib import Path

SAMPLE_JSON = Path('S:/workspace/model_custom/weights/sample_eval_set_20260311.json')
MODEL = Path('S:/workspace/model_custom/weights/custo_all.pt')
OUT = Path('S:/workspace/model_custom/weights/sample_eval_after_20260311.json')

data = json.loads(SAMPLE_JSON.read_text(encoding='utf-8'))
model = YOLO(str(MODEL))

out = {'model_path': str(MODEL), 'sample_count': len(data), 'items': []}
hits = 0
confs = []

for item in data:
    img = item['image']
    expected_id = item['class_id']
    expected_name = item['class_name']
    preds = model.predict(source=img, conf=0.001, device=0, imgsz=512, verbose=False)
    # preds is list length 1
    dets = []
    best_expected_conf = 0.0
    hit = False
    if len(preds) > 0 and getattr(preds[0], 'boxes', None) is not None:
        boxes = preds[0].boxes
        # boxes.cls, boxes.conf
        for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            dets.append({'class_id': int(cls), 'class_name': str(model.model.names[int(cls)]), 'conf': float(conf)})
            if int(cls) == expected_id:
                hit = True
                if float(conf) > best_expected_conf:
                    best_expected_conf = float(conf)
    if hit:
        hits += 1
        confs.append(best_expected_conf)
    out['items'].append({
        'image': img,
        'expected_class_id': expected_id,
        'expected_class_name': expected_name,
        'hit_expected_class': hit,
        'best_expected_conf': best_expected_conf,
        'detections': dets,
    })

out['expected_hits'] = hits
out['avg_conf_on_hits'] = sum(confs)/len(confs) if confs else 0.0
OUT.write_text(json.dumps(out, indent=2), encoding='utf-8')
print(f"Sample count: {len(data)} | Hits: {hits} | Avg conf: {out['avg_conf_on_hits']:.4f}")
print(f"Wrote: {OUT}")
