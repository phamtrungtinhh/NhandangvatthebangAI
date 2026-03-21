import json
import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    import app as appmod
except Exception:
    # fallback: load module by path
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "S:/workspace/app.py")
    appmod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(appmod)
    except Exception:
        # if loading fails, re-raise to surface error
        raise

MODEL = Path('S:/workspace/model_custom/weights/custo_all.pt')
IN_DIR = Path('S:/workspace/tmp_test_inputs')
OUT_DIR = Path('S:/workspace/tmp_test_outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / 'infer_log.json'

print("Mode: hybrid (running both custom + COCO models)")

# run sanity check on class order
try:
    check = appmod.check_class_order_and_warn()
    if check.get('mismatch'):
        print("WARNING: class order mismatch detected (see logs)")
except Exception:
    pass

custom_model = YOLO(str(MODEL))
coco_model = YOLO(str(appmod.COCO_MODEL_PATH))

summary = {'total_images': 0, 'total_detections': 0}
items = []

# optional single-image override
single_image = None
if len(sys.argv) > 1:
    single_image = Path(sys.argv[1])

paths = [single_image] if single_image else sorted(IN_DIR.rglob('*'))

for img_path in paths:
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

    # merged initial
    merged = custom_dets + coco_dets

    # counts helper
    def counts_from(dets):
        c = {}
        for d in dets or []:
            k = appmod.normalize_class_name(d.get('class', '')) or 'unknown'
            c[k] = c.get(k, 0) + 1
        return c

    raw_custom_counts = counts_from(custom_dets)
    raw_coco_counts = counts_from(coco_dets)

    # get image dims
    from PIL import Image as PILImage
    img = PILImage.open(img_path)
    w, h = img.size

    # after verify_and_reduce (use base_conf 0.35)
    after_verify = appmod.verify_and_reduce_detections(merged, img_w=w, img_h=h, base_conf=0.35)
    after_verify_counts = counts_from(after_verify)

    # finalize
    try:
        finalize_ret = appmod.finalize_frame_detections_for_count(after_verify, img_w=w, img_h=h, min_conf=0.35, debug=True)
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

    # save per-stage overlay images (and counts) if available
    try:
        if stage_dets:
            def draw_and_save(stage_name, dets):
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

            for sname, sd in (stage_dets or {}).items():
                if sd is None:
                    continue
                draw_and_save(sname, sd)
    except Exception:
        pass

    # save plotted image (from custom model visualization)
    try:
        im = r_c.plot()  # custom plot
        if isinstance(im, np.ndarray):
            Image.fromarray(im).save(OUT_DIR / img_path.name)
        else:
            Image.open(img_path).save(OUT_DIR / img_path.name)
    except Exception:
        Image.open(img_path).save(OUT_DIR / img_path.name)

    summary['total_detections'] += len(after_finalize)
    items.append({
        'image': str(img_path),
        'raw_custom_counts': raw_custom_counts,
        'raw_coco_counts': raw_coco_counts,
        'after_verify_counts': after_verify_counts,
        'after_finalize_counts': after_finalize_counts,
        'num_final': len(after_finalize),
        'stage_stats': stage_stats,
        'stage_detections_counts': {k: (len(v) if v else 0) for k, v in (stage_dets or {}).items()},
    })

out = {'summary': summary, 'items': items}
LOG.write_text(json.dumps(out, indent=2), encoding='utf-8')
print(f"Done. Images: {summary['total_images']}, final detections: {summary['total_detections']}")
print(f"Log: {LOG}")
