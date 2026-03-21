from ultralytics import YOLO
import json
from pathlib import Path

data = r'S:/workspace/model_custom/dataset/data.yaml'
models = [
    ('current', r'S:/workspace/model_custom/weights/custo_all.pt'),
    ('backup', r'S:/workspace/model_custom/weights/custo_all.pt.bak'),
]

out = {}
for name, path in models:
    p = Path(path)
    if not p.exists():
        print(f"skipping {name}, not found: {path}")
        continue
    print(f"running val for {name}: {path}")
    try:
        m = YOLO(str(path))
        metrics = m.val(data=data, imgsz=640, batch=2, device=0, workers=0, verbose=False)
    except Exception as e:
        print(f"error running val for {name}: {e}")
        out[name] = {'error': str(e)}
        continue
    # metrics.box may or may not have the expected attrs depending on ultralytics version
    box = getattr(metrics, 'box', None)
    tree_map = None
    try:
        if box is not None and hasattr(box, 'maps'):
            tree_map = box.maps[3]
    except Exception:
        tree_map = None
    out[name] = {
        'precision_all': float(box.mp) if box is not None and hasattr(box, 'mp') else None,
        'recall_all': float(box.mr) if box is not None and hasattr(box, 'mr') else None,
        'map50_all': float(box.map50) if box is not None and hasattr(box, 'map50') else None,
        'map_all': float(box.map) if box is not None and hasattr(box, 'map') else None,
        'map_tree': float(tree_map) if tree_map is not None else None,
    }

out_path = Path(r'S:/workspace/model_custom/weights/tree_val_compare.json')
out_path.write_text(json.dumps(out, indent=2))
print('wrote', out_path)
print(json.dumps(out, indent=2))
