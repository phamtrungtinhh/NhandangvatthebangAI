from ultralytics import YOLO
import json
from pathlib import Path

backup_bak = Path(r'S:/workspace/model_custom/weights/custo_all.pt.bak')
backup_pt = Path(r'S:/workspace/model_custom/weights/custo_all_backup.pt')
if not backup_bak.exists():
    print('backup .bak not found:', backup_bak)
    raise SystemExit(1)
# copy to safe .pt filename
backup_pt.write_bytes(backup_bak.read_bytes())
print('copied to', backup_pt)

m = YOLO(str(backup_pt))
metrics = m.val(data=r'S:/workspace/model_custom/dataset/data.yaml', imgsz=640, batch=2, device=0, workers=0, verbose=False)
box = getattr(metrics, 'box', None)
try:
    tree_map = box.maps[3] if box is not None and hasattr(box, 'maps') else None
except Exception:
    tree_map = None
out = {
    'precision_all': float(box.mp) if box is not None and hasattr(box, 'mp') else None,
    'recall_all': float(box.mr) if box is not None and hasattr(box, 'mr') else None,
    'map50_all': float(box.map50) if box is not None and hasattr(box, 'map50') else None,
    'map_all': float(box.map) if box is not None and hasattr(box, 'map') else None,
    'map_tree': float(tree_map) if tree_map is not None else None,
}

out_path = Path(r'S:/workspace/model_custom/weights/tree_val_compare.json')
existing = {}
if out_path.exists():
    try:
        existing = json.loads(out_path.read_text(encoding='utf-8'))
    except Exception:
        existing = {}
existing['backup'] = out
out_path.write_text(json.dumps(existing, indent=2))
print('wrote', out_path)
print(json.dumps(existing, indent=2))
