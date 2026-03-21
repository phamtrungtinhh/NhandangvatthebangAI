import sys
from pathlib import Path
from ultralytics import YOLO
import importlib.util
spec = importlib.util.spec_from_file_location("app", "S:/workspace/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

if len(sys.argv) < 2:
    print('Usage: trace_finalize_steps.py <image_path>')
    raise SystemExit(1)

img_path = Path(sys.argv[1])
MODEL = Path('S:/workspace/model_custom/weights/custo_all.pt')
custom_model = YOLO(str(MODEL))
coco_model = YOLO(str(app.COCO_MODEL_PATH))

# run models
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

from PIL import Image as PILImage
img = PILImage.open(img_path)
w, h = img.size

after_verify = app.verify_and_reduce_detections(merged, img_w=w, img_h=h, base_conf=0.35)
print('after_verify:', len(after_verify))

out = app.finalize_frame_detections_for_count.__wrapped__(after_verify, img_w=w, img_h=h, min_conf=0.35) if hasattr(app.finalize_frame_detections_for_count, '__wrapped__') else None
# Note: cannot easily call internal stepwise; instead replicate finalization chain manually

# start from merged_nms (per-class NMS)
dets = app.canonicalize_final_detections(after_verify)
# per-class dedup like finalize
grouped = {}
for d in dets:
    cls = app.normalize_class_name(d.get('class','')) or 'unknown'
    grouped.setdefault(cls, []).append(d)
merged_out = []
for cls, dets_cls in grouped.items():
    if cls == 'fruit' or cls in app.SPECIFIC_FRUIT_CLASSES:
        iou_thresh_cls = 0.72
    else:
        iou_thresh_cls = 0.64
    ordered = sorted(dets_cls, key=lambda x: float(x.get('conf',0.0)), reverse=True)
    kept = []
    for d in ordered:
        overlap = False
        for k in kept:
            if app._box_iou(d.get('xyxy'), k.get('xyxy')) >= float(iou_thresh_cls):
                overlap = True
                break
        if not overlap:
            kept.append(d)
    merged_out.extend(kept)

print('after_per_class_nms:', len(merged_out))
out = merged_out

# Define sequence of functions
sequence = [
    ('resolve_cross_class_overlaps', lambda x: app.resolve_cross_class_overlaps_with_priority(x, iou_thresh=0.60, conf_gap=0.06)),
    ('suppress_cross_class_overlaps', lambda x: app.suppress_cross_class_overlaps(x, iou_thresh=0.62, conf_gap=0.10)),
    ('suppress_generic_fruit_overlaps', lambda x: app.suppress_generic_fruit_overlaps(x, iou_thresh=0.35, min_specific_conf=0.55, conf_gap=0.05)),
    ('suppress_conflicting_specific_fruits', lambda x: app.suppress_conflicting_specific_fruits(x, iou_thresh=0.40, conf_gap=0.05)),
    ('suppress_scene_level_boxes', lambda x: app.suppress_scene_level_boxes(x, img_w=w, img_h=h)),
    ('enforce_one_object_one_box', lambda x: app.enforce_one_object_one_box(x, img_w=w, img_h=h)),
    ('normalize_dense_flower_boxes', lambda x: app.normalize_dense_flower_boxes(x, img_w=w, img_h=h)),
    ('collapse_dense_flower_duplicates', lambda x: app.collapse_dense_flower_duplicates(x, img_w=w, img_h=h)),
    ('prune_dense_flower_noise', lambda x: app.prune_dense_flower_noise(x, img_w=w, img_h=h, min_conf=0.35)),
    ('collapse_sparse_flower_duplicates', lambda x: app.collapse_sparse_flower_duplicates(x, img_w=w, img_h=h)),
    ('collapse_sparse_large_flower_duplicates', lambda x: app.collapse_sparse_large_flower_duplicates(x, img_w=w, img_h=h)),
    ('prune_dominant_flower_children', lambda x: app.prune_dominant_flower_children(x, img_w=w, img_h=h)),
    ('collapse_compact_flower_cluster', lambda x: app.collapse_compact_flower_cluster(x, img_w=w, img_h=h)),
    ('suppress_flower_cross_class_confusions', lambda x: app.suppress_flower_cross_class_confusions(x, img_w=w, img_h=h, base_conf=0.35)),
    ('dedup_detections_by_class_nms_classwise', lambda x: app.dedup_detections_by_class_nms_classwise(x, default_iou=0.70)),
]

current = out
for name, fn in sequence:
    new = fn(current)
    print(f"{name}: {len(new)}")
    current = new

print('final:', len(current))
# print removed flowers details

