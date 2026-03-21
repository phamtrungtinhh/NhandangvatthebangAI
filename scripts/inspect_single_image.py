import sys, json
from pathlib import Path
from ultralytics import YOLO

# Load app module by path
import importlib.util
spec = importlib.util.spec_from_file_location("app", "S:/workspace/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

if len(sys.argv) < 2:
    print('Usage: inspect_single_image.py <image_path>')
    raise SystemExit(1)

img_path = Path(sys.argv[1])
MODEL = Path('S:/workspace/model_custom/weights/custo_all.pt')
custom_model = YOLO(str(MODEL))
coco_model = YOLO(str(app.COCO_MODEL_PATH))

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

# helper to enrich detections with area/aspect
def enrich(dets, img_w, img_h):
    out = []
    for d in dets or []:
        xy = d.get('xyxy') or []
        if not xy or len(xy) < 4:
            continue
        a, asp, near_full = app._box_metrics(d.get('xyxy'), img_w, img_h)
        d2 = dict(d)
        d2['_area_ratio'] = a
        d2['_aspect'] = asp
        d2['_near_full'] = near_full
        out.append(d2)
    return out

from PIL import Image as PILImage
img = PILImage.open(img_path)
w, h = img.size

# after verify
after_verify = app.verify_and_reduce_detections(merged, img_w=w, img_h=h, base_conf=0.35)

# finalize with debug True to get stage snapshots
final_ret = app.finalize_frame_detections_for_count(after_verify, img_w=w, img_h=h, min_conf=0.35, debug=True)

# final_ret expected as (out, stage_stats, stage_dets)
out = None
stage_stats = None
stage_dets = None
if isinstance(final_ret, tuple):
    if len(final_ret) == 3:
        out, stage_stats, stage_dets = final_ret
    elif len(final_ret) == 2:
        out, stage_stats = final_ret
        stage_dets = {}
    else:
        out = final_ret[0]
        stage_stats = {}
        stage_dets = {}
else:
    out = final_ret
    stage_stats = {}
    stage_dets = {}

# enrich each stage dets
enriched = {}
for k, v in (stage_dets or {}).items():
    if v is None:
        enriched[k] = []
    else:
        enriched[k] = enrich(v, w, h)

# write to tmp_test_outputs
OUT = Path('S:/workspace/tmp_test_outputs')
OUT.mkdir(parents=True, exist_ok=True)
OUTF = OUT / 'inspect_stage_dets.json'
res = {'image': str(img_path), 'stage_stats': stage_stats, 'stage_detections': enriched}
OUTF.write_text(json.dumps(res, indent=2), encoding='utf-8')
print('Wrote', OUTF)
