import json
import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import importlib.util

# load app module
spec = importlib.util.spec_from_file_location("app", "S:/workspace/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

MODEL = Path('S:/workspace/model_custom/weights/custo_all.pt')
IN_DIR = Path('S:/workspace/tmp_test_inputs/selected')
OUT_DIR = Path('S:/workspace/tmp_test_outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / 'infer_log_selected.json'

custom_model = YOLO(str(MODEL))
coco_model = YOLO(str(app.COCO_MODEL_PATH))

items = []
summary = {'total_images': 0, 'total_detections': 0}

# helper
def counts_from(dets):
    c = {}
    for d in dets or []:
        k = app.normalize_class_name(d.get('class', '')) or 'unknown'
        c[k] = c.get(k, 0) + 1
    return c

def draw_and_save(img_path, stage_name, dets):
    try:
        im = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for d in (dets or []):
            xy = d.get('xyxy') or []
            if not xy or len(xy) < 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, xy[:4])
            except Exception:
                continue
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            clsn = str(d.get('class', ''))
            confv = float(d.get('conf', 0.0))
            text = f"{clsn}:{confv:.2f}"
            if font:
                draw.text((x1+2, y1+2), text, fill='yellow', font=font)
            else:
                draw.text((x1+2, y1+2), text, fill='yellow')
        out_name = OUT_DIR / f"{img_path.stem}__{stage_name}{img_path.suffix}"
        im.save(out_name)
    except Exception:
        pass

for img_path in sorted(IN_DIR.rglob('*')):
    if not img_path.is_file():
        continue
    summary['total_images'] += 1
    # run custom
    res_c = custom_model.predict(source=str(img_path), conf=0.20, imgsz=512, device=0, verbose=False)
    r_c = res_c[0]
    custom_dets = []
    if getattr(r_c, 'boxes', None) is not None and len(r_c.boxes) > 0:
        cls_list = r_c.boxes.cls.tolist()
        conf_list = r_c.boxes.conf.tolist()
        xyxy_list = [list(map(float, b.xyxy[0].cpu().numpy())) if hasattr(b, 'xyxy') else None for b in r_c.boxes]
        for idx, (cls, conf) in enumerate(zip(cls_list, conf_list)):
            cname = str(custom_model.model.names[int(cls)])
            xy = xyxy_list[idx] if idx < len(xyxy_list) else None
            custom_dets.append({'class': cname, 'conf': float(conf), 'xyxy': xy, '_source_model': 'custom'})

    # run coco
    res_co = coco_model.predict(source=str(img_path), conf=0.20, imgsz=512, device=0, verbose=False)
    r_co = res_co[0]
    coco_dets = []
    if getattr(r_co, 'boxes', None) is not None and len(r_co.boxes) > 0:
        cls_list = r_co.boxes.cls.tolist()
        conf_list = r_co.boxes.conf.tolist()
        xyxy_list = [list(map(float, b.xyxy[0].cpu().numpy())) if hasattr(b, 'xyxy') else None for b in r_co.boxes]
        for idx, (cls, conf) in enumerate(zip(cls_list, conf_list)):
            cname = str(coco_model.model.names[int(cls)])
            xy = xyxy_list[idx] if idx < len(xyxy_list) else None
            coco_dets.append({'class': cname, 'conf': float(conf), 'xyxy': xy, '_source_model': 'coco'})

    merged = custom_dets + coco_dets

    # get image dims
    from PIL import Image as PILImage
    img = PILImage.open(img_path)
    w, h = img.size

    after_verify = app.verify_and_reduce_detections(merged, img_w=w, img_h=h, base_conf=0.35)

    try:
        finalize_ret = app.finalize_frame_detections_for_count(after_verify, img_w=w, img_h=h, min_conf=0.35, debug=True)
        if isinstance(finalize_ret, tuple):
            if len(finalize_ret) == 3:
                after_finalize, stage_stats, stage_dets = finalize_ret
            elif len(finalize_ret) == 2:
                after_finalize, stage_stats = finalize_ret
                stage_dets = {}
            else:
                after_finalize = finalize_ret[0]
                stage_stats = {}
                stage_dets = {}
        else:
            after_finalize = finalize_ret
            stage_stats = {}
            stage_dets = {}
    except Exception:
        after_finalize = after_verify
        stage_stats = {}
        stage_dets = {}

    after_finalize_counts = counts_from(after_finalize)

    # save overlays for selected key stages
    for sname in ['after_flower_pruning', 'after_scene_rules']:
        sd = (stage_dets or {}).get(sname)
        if sd is not None:
            draw_and_save(img_path, sname, sd)

    summary['total_detections'] += len(after_finalize)
    items.append({
        'image': str(img_path),
        'raw_custom_counts': counts_from(custom_dets),
        'raw_coco_counts': counts_from(coco_dets),
        'after_verify_counts': counts_from(after_verify),
        'after_finalize_counts': after_finalize_counts,
        'num_final': len(after_finalize),
        'stage_stats': stage_stats,
        'stage_detections_counts': {k: (len(v) if v else 0) for k, v in (stage_dets or {}).items()},
    })

out = {'summary': summary, 'items': items}
LOG.write_text(json.dumps(out, indent=2), encoding='utf-8')

# aggregation
from statistics import mean
stages = ['after_nms', 'after_resolve_cross_class', 'after_flower_pruning', 'after_scene_rules', 'final']
per_image = []
for it in items:
    stats = it.get('stage_stats') or {}
    counts = {s: stats.get(s, 0) or 0 for s in stages}
    per_image.append({'image': it.get('image'), 'counts': counts, 'num_final': it.get('num_final', 0)})

print('Processed', summary['total_images'], 'images')
# stage summaries
for s in stages:
    vals = [p['counts'][s] for p in per_image]
    if vals:
        print(f"- {s}: mean={mean(vals):.2f}, min={min(vals)}, max={max(vals)}")
    else:
        print(f"- {s}: no data")

# top-5 drops
drops = []
for p in per_image:
    c = p['counts']
    prev = None
    prev_stage = None
    max_drop = 0
    max_pair = None
    for s in stages:
        if prev is not None:
            drop = prev - c[s]
            if drop > max_drop:
                max_drop = drop
                max_pair = (prev_stage, s)
        prev = c[s]
        prev_stage = s
    drops.append({'image': p['image'], 'max_drop': max_drop, 'pair': max_pair, 'counts': c})

sorted_drops = sorted(drops, key=lambda x: x['max_drop'], reverse=True)
print('\nTop-5 images with largest single-stage drops:')
for d in sorted_drops[:5]:
    print(f"- {d['image']}: drop={d['max_drop']} at {d['pair']}, counts={d['counts']}")

# fallback_used count
fallbacks = sum(1 for it in items if (it.get('stage_stats') or {}).get('fallback_used'))
print('\nFallback_used count:', fallbacks)

# optional baseline comparison
BASE = OUT_DIR / 'infer_log_prepatch.json'
if BASE.exists():
    old = json.loads(BASE.read_text(encoding='utf-8'))
    old_map = {it['image']: it.get('num_final', 0) for it in old.get('items', [])}
    changed = 0
    for it in items:
        img = it['image']
        oldv = old_map.get(img)
        newv = it.get('num_final', 0)
        if oldv is not None and oldv != newv:
            changed += 1
            print(f"Image {img} changed final {oldv} -> {newv}")
    print('Images with final changed vs baseline:', changed)
else:
    print('No baseline file found at', BASE)
