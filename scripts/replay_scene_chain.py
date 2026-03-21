import json
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location("app", "S:/workspace/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

F = Path('S:/workspace/tmp_test_outputs/inspect_stage_dets.json')
if not F.exists():
    print('inspect_stage_dets.json missing')
    raise SystemExit(1)

data = json.loads(F.read_text(encoding='utf-8'))
stage_dets = data.get('stage_detections', {})
start = stage_dets.get('after_flower_pruning', [])
print('start count:', len(start))
from PIL import Image as PILImage
img_path = data.get('image')
if img_path:
    im = PILImage.open(img_path)
    w, h = im.size
else:
    w, h = 0, 0

chain = [
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
]

current = start
for name, fn in chain:
    try:
        out = fn(current)
    except Exception as e:
        out = current
    print(f"{name}: {len(out)}")
    current = out

print('result count:', len(current))
print('final list classes:', [d.get('class') for d in current])
