import streamlit as st
import sys
import os
print("PYTHON:", sys.executable)
print("CWD:", os.getcwd())
import sqlite3
import json
import streamlit as st
import sqlite3
import json
import os
import tempfile
from datetime import datetime
import requests

from PIL import Image
import io
import numpy as np
import cv2

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

DB_PATH = "analysis.db"
IMG_SIZE_OPTIONS = [320, 416, 512, 640, 768, 1024]
CUSTOM_MODEL_PATH = "models/custom_all.pt"
FLOWER_SPECIALIST_MODEL_PATH = "models/flower_best.pt"
DUMBBELL_SPECIALIST_MODEL_PATH = "models/dumbbell_best.pt"
FRUIT_SPECIALIST_MODEL_PATH = "models/fruit_best.pt"
TREE_SPECIALIST_MODEL_PATH = "models/tree_best.pt"
FLOWER_MODEL_PATH = CUSTOM_MODEL_PATH
COCO_MODEL_PATH = "yolo11s.pt"
ENABLE_DENSE_FLOWER_FALLBACK = False


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp TEXT,
            media_type TEXT,
            objects_json TEXT
        )
        """
    )
    # ensure older DBs get the media_type column if missing
    try:
        cur.execute("PRAGMA table_info(analyses)")
        cols = [r[1] for r in cur.fetchall()]
        if 'media_type' not in cols:
            cur.execute("ALTER TABLE analyses ADD COLUMN media_type TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()


@st.cache_resource
def load_model(model_name: str = "yolo11s.pt"):
    if YOLO is None:
        raise RuntimeError("`ultralytics` package is not available. Install requirements.")
    # if model_name is a path to an existing file, use it directly
    # Resolve whether model_name is a local path or a model identifier
    try:
        if os.path.exists(model_name):
            path_or_name = model_name
        else:
            # if user provided something that looks like a file but it doesn't exist, raise a clear error
            if model_name.lower().endswith('.pt') or os.sep in model_name or '/' in model_name:
                raise RuntimeError(f"Model weights not found at path: {model_name}. Upload the file vào workspace hoặc bỏ trống để dùng model COCO.")
            # if user typed short name like "yolo11s", append .pt so ultralytics will auto-download
            path_or_name = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
    except RuntimeError:
        raise
    except Exception:
        path_or_name = model_name
    model = YOLO(path_or_name)
    # move model to GPU if available for faster inference
    try:
        if torch is not None and torch.cuda.is_available():
            model.to('cuda')
    except Exception:
        pass
    return model


def recommend_params(file_bytes: bytes, is_image: bool):
    """Auto-tune confidence and imgsz for better accuracy with low false positives."""
    conf = 0.35  # start slightly higher than 0.1 to cut false positives
    imgsz = 640
    if file_bytes:
        try:
            if is_image:
                im = Image.open(io.BytesIO(file_bytes))
                w, h = im.size
                max_dim = max(w, h)
                imgsz = min(1024, max_dim)
            else:
                imgsz = 640
        except Exception:
            pass
    # snap to nearest allowed size
    imgsz = min(IMG_SIZE_OPTIONS, key=lambda x: abs(x - imgsz))
    return conf, imgsz


def auto_tune_thresholds(img: np.ndarray, raw_custom: dict, coco_dets: list, merged_special_dets: list):
    """Return tuned overlay_conf_thresh, per_class_thresholds dict, min_area_pct, require_consensus
    Strategy:
    - If many raw detections and low average confidence -> raise thresholds
    - If very few detections -> lower thresholds slightly to allow recall
    - For crowded scenes (many small boxes) increase min_area_pct
    - Always prefer requiring consensus when specialists available
    """
    h, w = img.shape[:2]
    area = float(max(1, w * h))
    raw_list = raw_custom.get('detections', []) or []
    n_raw = len(raw_list)
    avg_conf = (sum([float(d.get('conf', 0.0)) for d in raw_list]) / n_raw) if n_raw > 0 else 0.0
    avg_area = 0.0
    if n_raw:
        areas = []
        for d in raw_list:
            xy = d.get('xyxy')
            if xy and len(xy) >= 4:
                bw = max(0.0, float(xy[2]) - float(xy[0]))
                bh = max(0.0, float(xy[3]) - float(xy[1]))
                areas.append((bw * bh) / area)
        avg_area = sum(areas) / len(areas) if areas else 0.0

    # base values from session (fallback)
    base_overlay = float(st.session_state.get('flower_overlay_conf', 0.35))
    base_min_area_pct = float(st.session_state.get('flower_area_pct', 0.20))
    per_class = {
        'dumbbell': float(st.session_state.get('th_dumbbell', 0.40)),
        'flower': float(st.session_state.get('th_flower', 0.40)),
        'fruit': float(st.session_state.get('th_fruit', 0.40)),
        'tree': float(st.session_state.get('th_tree', 0.40)),
    }

    # Start tuning
    overlay = base_overlay
    min_area_pct = base_min_area_pct
    require_cons = True if merged_special_dets else True

    # If many raw detections relative to image area, tighten a lot
    density = n_raw / max(1.0, (w * h) / (256.0 * 256.0))
    if n_raw >= 12 or density > 6.0:
        overlay = max(0.65, base_overlay + 0.25)
        for k in per_class:
            per_class[k] = max(per_class[k], 0.70)
        min_area_pct = max(min_area_pct, 0.30)
    elif n_raw >= 6 or density > 3.0:
        overlay = max(0.55, base_overlay + 0.15)
        for k in per_class:
            per_class[k] = max(per_class[k], 0.60)
        min_area_pct = max(min_area_pct, 0.20)
    else:
        # few detections -> be permissive but still avoid whole-image scans
        if avg_conf < 0.35 and n_raw > 0:
            overlay = min(0.60, base_overlay + 0.10)
        elif avg_conf > 0.70:
            overlay = max(base_overlay, 0.35)

    # if avg box area tiny -> increase min_area_pct
    if avg_area > 0 and avg_area < 0.0005:
        min_area_pct = max(min_area_pct, 0.40)
    elif avg_area > 0 and avg_area < 0.0015:
        min_area_pct = max(min_area_pct, 0.25)

    # Cap values
    overlay = min(0.95, overlay)
    for k in per_class:
        per_class[k] = min(0.95, per_class[k])

    return overlay, per_class, min_area_pct, require_cons


# Vietnamese label translations (english -> vietnamese) limited to COCO 80 classes
BASE_VN_LABELS = {
    'person': 'người',
    'bicycle': 'xe đạp',
    'car': 'ô tô',
    'motorcycle': 'xe máy',
    'airplane': 'máy bay',
    'bus': 'xe buýt',
    'train': 'tàu hỏa',
    'truck': 'xe tải',
    'boat': 'thuyền',
    'traffic light': 'đèn giao thông',
    'fire hydrant': 'vòi cứu hỏa',
    'stop sign': 'biển dừng',
    'parking meter': 'đồng hồ đỗ xe',
    'bench': 'ghế băng',
    'bird': 'chim',
    'cat': 'mèo',
    'dog': 'chó',
    'horse': 'ngựa',
    'sheep': 'cừu',
    'cow': 'bò',
    'elephant': 'voi',
    'bear': 'gấu',
    'zebra': 'ngựa vằn',
    'giraffe': 'hươu cao cổ',
    'backpack': 'ba lô',
    'umbrella': 'ô',
    'handbag': 'túi xách',
    'tie': 'cà vạt',
    'suitcase': 'vali',
    'frisbee': 'đĩa ném',
    'skis': 'ván trượt tuyết',
    'snowboard': 'ván trượt tuyết',
    'sports ball': 'bóng',
    'kite': 'diều',
    'baseball bat': 'gậy bóng chày',
    'baseball glove': 'găng bóng chày',
    'skateboard': 'ván trượt',
    'surfboard': 'ván lướt',
    'tennis racket': 'vợt tennis',
    'bottle': 'chai',
    'wine glass': 'ly rượu',
    'cup': 'cốc',
    'fork': 'nĩa',
    'knife': 'dao',
    'spoon': 'thìa',
    'bowl': 'bát',
    'banana': 'chuối',
    'apple': 'táo',
    'sandwich': 'bánh mì kẹp',
    'orange': 'cam',
    'broccoli': 'bông cải',
    'carrot': 'cà rốt',
    'hot dog': 'xúc xích',
    'pizza': 'pizza',
    'donut': 'bánh vòng',
    'cake': 'bánh kem',
    'chair': 'ghế',
    'couch': 'ghế sofa',
    'potted plant': 'cây trong chậu',
    'bed': 'giường',
    'dining table': 'bàn ăn',
    'toilet': 'bồn cầu',
    'tv': 'tivi',
    'laptop': 'máy tính xách tay',
    'mouse': 'chuột',
    'remote': 'điều khiển',
    'keyboard': 'bàn phím',
    'cell phone': 'điện thoại',
    'microwave': 'lò vi sóng',
    'oven': 'lò nướng',
    'toaster': 'máy nướng bánh',
    'sink': 'bồn rửa',
    'refrigerator': 'tủ lạnh',
    'book': 'sách',
    'clock': 'đồng hồ',
    'vase': 'lọ hoa',
    'scissors': 'kéo',
    'teddy bear': 'gấu bông',
    'hair drier': 'máy sấy tóc',
    'toothbrush': 'bàn chải đánh răng'
}

# Custom classes
BASE_VN_LABELS['flower'] = 'hoa'
BASE_VN_LABELS['tree'] = 'cây'
BASE_VN_LABELS['fruit'] = 'trái cây'
BASE_VN_LABELS['dumbbell'] = 'tạ tay'

# Extra labels (kept empty to remain at 80 COCO classes)
EXTRA_VN_LABELS = {}

# Final merged mapping: keep base labels and add extras only if key missing


# Final merged mapping: keep base labels and add extras only if key missing
VN_LABELS = dict(BASE_VN_LABELS)
for _k, _v in EXTRA_VN_LABELS.items():
    if _k not in VN_LABELS:
        VN_LABELS[_k] = _v


def update_vn_labels(new_entries: dict, overwrite: bool = False):
    """Merge new_entries into VN_LABELS.

    If overwrite is False, existing keys are preserved. If True, new values replace existing ones.
    """
    for k, v in (new_entries or {}).items():
        if overwrite or k not in VN_LABELS:
            VN_LABELS[k] = v


def translate_label(name: str) -> str:
    if not name:
        return name
    key = name.lower().strip()
    # normalize common separators
    key = key.replace('_', ' ').replace('-', ' ')

    # direct match
    if key in VN_LABELS:
        return VN_LABELS[key]

    # try simple singularization (strip trailing s or es)
    if key.endswith('es'):
        k2 = key[:-2]
        if k2 in VN_LABELS:
            return VN_LABELS[k2]
    if key.endswith('s'):
        k2 = key[:-1]
        if k2 in VN_LABELS:
            return VN_LABELS[k2]

    # common aliases mapping for animals and vehicles
    ALIASES = {
        'ox': 'cow',
        'cattle': 'cow',
        'puppy': 'dog',
        'kitten': 'cat',
        'motorbike': 'motorcycle',
        'lorry': 'truck',
        'sofa': 'couch',
        'tvmonitor': 'tv',
    }
    if key in ALIASES and ALIASES[key] in VN_LABELS:
        return VN_LABELS[ALIASES[key]]

    # try split and match last token (e.g., 'sports ball' -> 'ball')
    parts = key.split()
    for p in reversed(parts):
        if p in VN_LABELS:
            return VN_LABELS[p]

    # fallback: return original name (preserve capitalization as provided)
    return name


def normalize_class_name(name: str) -> str:
    if not name:
        return ""
    key = str(name).lower().strip().replace('_', ' ').replace('-', ' ')
    alias = {
        'hoa': 'flower',
        'bong hoa': 'flower',
        'flower': 'flower',
        'cay': 'tree',
        'cây': 'tree',
        'tree': 'tree',
        'trai cay': 'fruit',
        'trái cây': 'fruit',
        'hoa qua': 'fruit',
        'fruit': 'fruit',
        'ta tay': 'dumbbell',
        'tạ tay': 'dumbbell',
        'dumbbell': 'dumbbell',
    }
    return alias.get(key, key)


def canonicalize_final_detections(detections: list) -> list:
    out = []
    for d in detections or []:
        if not isinstance(d, dict):
            continue
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        try:
            confv = float(d.get('conf', 0.0))
        except Exception:
            confv = 0.0
        xyxy = d.get('xyxy') if isinstance(d.get('xyxy'), (list, tuple)) else None
        out.append({
            'class': cls_name,
            'conf': confv,
            'xyxy': list(xyxy[:4]) if xyxy and len(xyxy) >= 4 else xyxy,
        })
    return out


def build_counts_from_detections(final_detections: list) -> dict:
    counts = {}
    for d in final_detections or []:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts


def extract_final_detections_from_raw(raw: dict) -> list:
    if not isinstance(raw, dict):
        return []
    if isinstance(raw.get('detections'), list):
        return canonicalize_final_detections(raw.get('detections') or [])
    if isinstance(raw.get('frames'), list):
        frame_dets = []
        for f in raw.get('frames') or []:
            if not isinstance(f, dict):
                continue
            frame_dets.append({
                'class': f.get('class', ''),
                'conf': f.get('conf', 0.0),
                'xyxy': f.get('xyxy'),
            })
        return canonicalize_final_detections(frame_dets)
    return []


def verify_and_reduce_detections(final_detections: list, img_w: int, img_h: int, base_conf: float = 0.25, strict_coco: bool = False) -> list:
    custom_classes = {'flower', 'fruit', 'tree', 'dumbbell'}
    dets = canonicalize_final_detections(final_detections)
    filtered = []

    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        confv = float(d.get('conf', 0.0))
        area_ratio, aspect, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)

        is_custom = cls_name in custom_classes
        if is_custom:
            min_conf = max(0.16, float(base_conf) * 0.70)
        else:
            min_conf = max(0.32, float(base_conf) + (0.18 if strict_coco else 0.08))

        if confv < min_conf:
            continue
        if area_ratio <= 0:
            continue
        if area_ratio < 0.00035 and confv < 0.75:
            continue
        # keep near-full boxes for custom classes with moderate confidence;
        # only apply stricter rule to non-custom classes to reduce COCO noise.
        if near_full:
            if is_custom and confv < 0.68:
                continue
            if (not is_custom) and confv < 0.86:
                continue
            continue
        if aspect > 10.0 and confv < 0.90:
            continue

        d2 = dict(d)
        d2['class'] = cls_name
        filtered.append(d2)

    # Class-aware NMS to reduce same-class duplicates
    filtered = dedup_detections_by_class_nms_classwise(filtered, default_iou=0.50)

    # Cross-class suppression: if 2 classes heavily overlap, keep stronger one
    ordered = sorted(filtered, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    kept = []
    for d in ordered:
        drop = False
        for k in kept:
            if str(d.get('class', '')) == str(k.get('class', '')):
                continue
            iou = _box_iou(d.get('xyxy'), k.get('xyxy'))
            if iou >= 0.78 and float(d.get('conf', 0.0)) <= float(k.get('conf', 0.0)) + 0.10:
                drop = True
                break
        if not drop:
            kept.append(d)

    # Guardrail: prevent runaway counts from noisy classes
    per_class_cap = 60
    out = []
    per_class_count = {}
    for d in kept:
        c = str(d.get('class', 'unknown')) or 'unknown'
        n = per_class_count.get(c, 0)
        if n >= per_class_cap:
            continue
        per_class_count[c] = n + 1
        out.append(d)
    return out


def annotate_image(img: np.ndarray, results, names_map=None, conf_thresh=0.25):
    img = img.copy()
    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
            cls = int(box.cls[0]) if hasattr(box, 'cls') else None
            if conf < conf_thresh:
                continue
            if not hasattr(box, 'xyxy'):
                continue
            coords = box.xyxy[0]
            # handle torch tensor or numpy
            try:
                coords = coords.cpu().numpy()
            except Exception:
                coords = np.array(coords)
            if coords is None or coords.shape[0] < 4:
                continue
            if not np.all(np.isfinite(coords)):
                continue
            x1, y1, x2, y2 = map(int, coords[:4])
            # original (english) label from model
            label = names_map.get(cls, str(cls)) if names_map else str(cls)
            label_norm = normalize_class_name(label)
            # display Vietnamese label when possible, fallback to 'Unknown'
            display_label = translate_label(label_norm) if label_norm else 'Unknown'
            # draw box + readable label background
            color = _color_for_label(str(label_norm or label or 'unknown'))
            _draw_box_with_label(img, (x1, y1, x2, y2), f"{display_label} {conf:.2f}", color)
    return img


def run_detection_on_image(model, image_np, conf: float, imgsz: int = 640, use_fp16: bool = True):
    predict_kwargs = dict(source=image_np, conf=conf, imgsz=imgsz)
    # use half precision if requested and CUDA available
    if use_fp16 and (torch is not None and torch.cuda.is_available()):
        predict_kwargs['half'] = True
    try:
        predict_kwargs['iou'] = float(st.session_state.get('nms_iou', 0.7))
        predict_kwargs['max_det'] = int(st.session_state.get('max_det', 300))
    except Exception:
        pass
    results = model.predict(**predict_kwargs)
    names = results[0].names if hasattr(results[0], 'names') else {}
    ann = annotate_image(image_np, results, names, conf_thresh=conf)
    counts = {}
    detections = []
    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        for box in boxes:
            confv = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
            if confv < conf:
                continue
            cls = int(box.cls[0]) if hasattr(box, 'cls') else None
            name = normalize_class_name(names.get(cls, str(cls)))
            counts[name] = counts.get(name, 0) + 1
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else None
            detections.append({
                "class": name,
                "conf": float(confv),
                "xyxy": xyxy.tolist() if xyxy is not None else None,
            })
    return ann, counts, detections


def run_tiled_flower_detections(model, image_np: np.ndarray, base_conf: float, imgsz: int = 640, use_fp16: bool = True):
    """Fallback for dense flower scenes: run inference on overlapping tiles.
    This helps recover multiple instances when full-image inference collapses to one large flower box.
    """
    if image_np is None or image_np.size == 0:
        return []

    h, w = image_np.shape[:2]
    if h < 64 or w < 64:
        return []

    tile_w = max(256, int(w * 0.62))
    tile_h = max(256, int(h * 0.62))
    stride_x = max(96, int(tile_w * 0.75))
    stride_y = max(96, int(tile_h * 0.75))

    xs = list(range(0, max(1, w - tile_w + 1), stride_x))
    ys = list(range(0, max(1, h - tile_h + 1), stride_y))
    if not xs or xs[-1] != max(0, w - tile_w):
        xs.append(max(0, w - tile_w))
    if not ys or ys[-1] != max(0, h - tile_h):
        ys.append(max(0, h - tile_h))

    tile_conf = max(0.05, float(base_conf) * 0.35)
    out = []
    for y0 in ys:
        for x0 in xs:
            tile = image_np[y0:y0 + tile_h, x0:x0 + tile_w]
            if tile.size == 0:
                continue
            predict_kwargs = dict(source=tile, conf=tile_conf, imgsz=imgsz)
            if use_fp16 and (torch is not None and torch.cuda.is_available()):
                predict_kwargs['half'] = True
            try:
                predict_kwargs['iou'] = 0.85
                predict_kwargs['max_det'] = 500
            except Exception:
                pass

            results = model.predict(**predict_kwargs)
            names = results[0].names if hasattr(results[0], 'names') else {}
            for r in results:
                boxes = getattr(r, 'boxes', None)
                if boxes is None:
                    continue
                for box in boxes:
                    confv = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else None
                    name = normalize_class_name(names.get(cls, str(cls)))
                    if name != 'flower':
                        continue
                    if not hasattr(box, 'xyxy'):
                        continue
                    coords = box.xyxy[0]
                    try:
                        coords = coords.cpu().numpy()
                    except Exception:
                        coords = np.array(coords)
                    if coords is None or coords.shape[0] < 4 or not np.all(np.isfinite(coords[:4])):
                        continue
                    x1, y1, x2, y2 = map(float, coords[:4])
                    out.append({
                        'class': 'flower',
                        'conf': float(confv),
                        'xyxy': [x1 + x0, y1 + y0, x2 + x0, y2 + y0],
                    })

    if not out:
        return []
    return dedup_detections_by_class_nms_classwise(out, default_iou=0.92)


def refine_custom_detections_with_specialists(image_np: np.ndarray, detections: list, base_conf: float, imgsz: int = 640, use_fp16: bool = True):
    """Refine custom classes using specialist models to reduce false positives and recover misses.
    - Keep high-confidence custom detections directly.
    - Require specialist overlap for lower-confidence custom detections.
    - Add specialist detections not already covered.
    """
    if image_np is None or image_np.size == 0:
        return canonicalize_final_detections(detections)

    custom_classes = ('flower', 'fruit', 'tree', 'dumbbell')
    model_map = {
        'flower': FLOWER_SPECIALIST_MODEL_PATH,
        'fruit': FRUIT_SPECIALIST_MODEL_PATH,
        'tree': TREE_SPECIALIST_MODEL_PATH,
        'dumbbell': DUMBBELL_SPECIALIST_MODEL_PATH,
    }
    high_conf = {
        'flower': 0.78,
        'fruit': 0.74,
        'tree': 0.74,
        'dumbbell': 0.70,
    }
    match_iou = {
        'flower': 0.30,
        'fruit': 0.35,
        'tree': 0.35,
        'dumbbell': 0.30,
    }

    base = canonicalize_final_detections(detections)
    base = [d for d in base if (normalize_class_name(d.get('class', '')) or 'unknown') in custom_classes]

    specialists = {c: [] for c in custom_classes}
    for cls_name in custom_classes:
        path = model_map.get(cls_name)
        if not path or not os.path.exists(path):
            continue
        try:
            specialist_model = load_model(path)
            _, _, dets = run_detection_on_image(
                specialist_model,
                image_np,
                conf=float(max(0.15, float(base_conf) * 0.70)),
                imgsz=imgsz,
                use_fp16=use_fp16,
            )
            dets = normalize_specialist_detections(dets, cls_name)
            specialists[cls_name] = canonicalize_final_detections(dets)
        except Exception:
            specialists[cls_name] = []

    refined = []
    for d in base:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        confv = float(d.get('conf', 0.0))
        if cls_name not in custom_classes:
            continue

        if confv >= high_conf[cls_name]:
            refined.append(d)
            continue

        spec = specialists.get(cls_name, [])
        if not spec:
            # no specialist evidence -> require stronger confidence to keep
            if confv >= max(high_conf[cls_name], float(base_conf) + 0.30):
                refined.append(d)
            continue

        if any(_box_iou(d.get('xyxy'), s.get('xyxy')) >= match_iou[cls_name] for s in spec):
            refined.append(d)

    # Recover misses from specialists (especially dumbbell/tree)
    for cls_name in custom_classes:
        for s in specialists.get(cls_name, []):
            overlap = any(
                (normalize_class_name(r.get('class', '')) == cls_name) and _box_iou(r.get('xyxy'), s.get('xyxy')) >= 0.40
                for r in refined
            )
            if not overlap:
                refined.append(s)

    refined = dedup_detections_by_class_nms_classwise(refined, default_iou=0.55)
    return refined


def run_detection_on_video(model, video_bytes, conf: float, imgsz: int = 640, use_fp16: bool = True, max_frames=30, sample_rate=5, batch_size: int = 4):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmp.write(video_bytes)
    tmp.flush()
    tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    frame_idx = 0
    aggregated = {}
    first_annotated = None
    frames_info = []
    frames_batch = []
    frames_batch_idx = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_batch.append(img)
            frames_batch_idx.append(int(frame_idx))

            # Run batch inference when batch is full
            if len(frames_batch) >= batch_size:
                predict_kwargs = dict(source=frames_batch, conf=conf, imgsz=imgsz)
                if use_fp16 and (torch is not None and torch.cuda.is_available()):
                    predict_kwargs['half'] = True
                try:
                    # pass NMS IoU and max_det when available (tunable in UI)
                    predict_kwargs['iou'] = float(st.session_state.get('nms_iou', 0.7))
                    predict_kwargs['max_det'] = int(st.session_state.get('max_det', 300))
                except Exception:
                    pass
                results_batch = model.predict(**predict_kwargs)
                for i, results in enumerate(results_batch):
                    names = results[0].names if hasattr(results[0], 'names') else {}
                    fidx = frames_batch_idx[i]
                    for r in results:
                        boxes = getattr(r, 'boxes', None)
                        if boxes is None:
                            continue
                        for box in boxes:
                            confv = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                            if confv < conf:
                                continue
                            cls = int(box.cls[0]) if hasattr(box, 'cls') else None
                            name = normalize_class_name(names.get(cls, str(cls)))
                            aggregated[name] = aggregated.get(name, 0) + 1
                            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else None
                            frames_info.append({
                                "frame": int(fidx),
                                "class": name,
                                "conf": float(confv),
                                "xyxy": xyxy.tolist() if xyxy is not None else None,
                            })
                    if first_annotated is None:
                        raw_img = frames_batch[i]
                        first_annotated = annotate_image(raw_img, results, names, conf_thresh=conf)
                frames_batch = []
                frames_batch_idx = []
        frame_idx += 1
        if frame_idx >= max_frames:
            break
    cap.release()
    try:
        os.unlink(tmp.name)
    except Exception:
        pass

    # process remaining frames in batch
    if frames_batch:
        predict_kwargs = dict(source=frames_batch, conf=conf, imgsz=imgsz)
        if use_fp16 and (torch is not None and torch.cuda.is_available()):
            predict_kwargs['half'] = True
        try:
            predict_kwargs['iou'] = float(st.session_state.get('nms_iou', 0.7))
            predict_kwargs['max_det'] = int(st.session_state.get('max_det', 300))
        except Exception:
            pass
        results_batch = model.predict(**predict_kwargs)
        for i, results in enumerate(results_batch):
            names = results[0].names if hasattr(results[0], 'names') else {}
            fidx = frames_batch_idx[i]
            for r in results:
                boxes = getattr(r, 'boxes', None)
                if boxes is None:
                    continue
                for box in boxes:
                    confv = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    if confv < conf:
                        continue
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else None
                    name = normalize_class_name(names.get(cls, str(cls)))
                    aggregated[name] = aggregated.get(name, 0) + 1
                    xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else None
                    frames_info.append({
                        "frame": int(fidx),
                        "class": name,
                        "conf": float(confv),
                        "xyxy": xyxy.tolist() if xyxy is not None else None,
                    })
            if first_annotated is None:
                raw_img = frames_batch[i]
                first_annotated = annotate_image(raw_img, results, names, conf_thresh=conf)

    if first_annotated is None:
        first_annotated = np.zeros((480, 640, 3), dtype=np.uint8)
    return first_annotated, aggregated, frames_info


def process_uploaded(model, uploaded_bytes: bytes, is_image: bool, conf: float, imgsz: int = 640, use_fp16: bool = True, max_frames: int = 30, sample_rate: int = 5, batch_size: int = 4, nms_iou: float = None, max_det: int = None):
    if is_image:
        pil = Image.open(io.BytesIO(uploaded_bytes))
        img = np.array(pil.convert('RGB'))
        # pass through NMS/io settings if provided via kwargs or session
        ann, counts, detections = run_detection_on_image(model, img, conf, imgsz=imgsz, use_fp16=use_fp16)
        detections = canonicalize_final_detections(detections)

        # Dense-flower fallback: if full-image inference collapses to one large flower box,
        # run tiled inference and merge to improve object counting.
        img_h, img_w = img.shape[:2]
        flower_dets = [d for d in detections if str(d.get('class', '')).lower().strip() == 'flower']
        has_large_flower_box = False
        for d in flower_dets:
            area_ratio, _, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)
            if near_full or area_ratio >= 0.38:
                has_large_flower_box = True
                break

        if st.session_state.get("enable_dense_flower_fallback", ENABLE_DENSE_FLOWER_FALLBACK) and len(flower_dets) <= 1 and has_large_flower_box:
            tiled_flower = run_tiled_flower_detections(
                model,
                img,
                base_conf=float(conf),
                imgsz=imgsz,
                use_fp16=use_fp16,
            )
            if len(tiled_flower) > len(flower_dets):
                merged = detections + tiled_flower
                detections = dedup_detections_by_class_nms_classwise(merged, default_iou=0.92)

        # keep display and metrics consistent: filter detections once, then redraw and recount
        filtered = []
        for d in detections or []:
            d2 = dict(d)
            d2['class'] = normalize_class_name(d2.get('class', '')) or 'unknown'
            confv = float(d2.get('conf', 0.0))
            area_ratio, _, near_full = _box_metrics(d2.get('xyxy'), img_w, img_h)
            if area_ratio <= 0:
                continue
            if near_full and confv < 0.70:
                continue
            if d2['class'] == 'flower' and area_ratio > 0.92 and confv < 0.60:
                continue
            filtered.append(d2)

        pre_verified = dedup_detections_by_class_nms_classwise(filtered, default_iou=0.60)
        filtered = verify_and_reduce_detections(
            filtered,
            img_w=img_w,
            img_h=img_h,
            base_conf=float(conf),
            strict_coco=False,
        )

        # Root-fix: never let strict post-filtering erase all valid detections.
        if not filtered and pre_verified:
            filtered = pre_verified

        # Last-resort fallback from raw detections with permissive confidence.
        if not filtered and detections:
            fallback = []
            min_conf = max(0.08, float(conf) * 0.45)
            for d in canonicalize_final_detections(detections):
                confv = float(d.get('conf', 0.0))
                if confv < min_conf:
                    continue
                area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
                if area_ratio <= 0:
                    continue
                fallback.append(d)
            filtered = dedup_detections_by_class_nms_classwise(fallback, default_iou=0.65)

        counts = {}
        for d in filtered:
            c = str(d.get('class', '')).lower().strip() or 'unknown'
            counts[c] = counts.get(c, 0) + 1
        ann = overlay_detections(img, filtered, conf_thresh=float(max(0.10, conf * 0.60)))
        return ann, counts, {"detections": filtered}, img
    else:
        ann, counts, frames_info = run_detection_on_video(model, uploaded_bytes, conf, imgsz=imgsz, use_fp16=use_fp16, max_frames=max_frames, sample_rate=sample_rate, batch_size=batch_size)
        return ann, counts, {"frames": frames_info}, None


def overlay_detections(img: np.ndarray, detections: list, conf_thresh: float, color=(255, 0, 255)):
    out = img.copy()
    for d in detections or []:
        confv = float(d.get("conf", 0.0))
        if confv < conf_thresh:
            continue
        xyxy = d.get("xyxy")
        if not xyxy or len(xyxy) < 4:
            continue
        x1, y1, x2, y2 = map(int, xyxy[:4])
        label = str(d.get("class", ""))
        disp = translate_label(label) if label else 'Unknown'
        col = _color_for_label(label or str(confv))
        _draw_box_with_label(out, (x1, y1, x2, y2), f"{disp} {confv:.2f}", col)
    return out


def _color_for_label(name: str):
    try:
        if not name:
            name = 'obj'
        s = sum(ord(c) for c in name)
        r = 100 + (s * 37) % 156
        g = 100 + (s * 57) % 156
        b = 100 + (s * 97) % 156
        return (int(b), int(g), int(r))
    except Exception:
        return (255, 0, 255)


def _draw_box_with_label(img: np.ndarray, xy, text: str, color=(255, 0, 255)):
    x1, y1, x2, y2 = map(int, xy)
    h, w = img.shape[:2]
    # clamp
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    # ensure box is valid
    if x2 <= x1 or y2 <= y1:
        return
    # draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # text params scaled by box height
    box_h = max(12, y2 - y1)
    font_scale = max(0.4, min(1.2, box_h / 200.0))
    thickness = 1 if font_scale < 0.7 else 2
    ((text_w, text_h), baseline) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = max(4, int(text_h * 0.3))
    # prefer above box unless not enough space
    txt_x1 = x1
    txt_y1 = y1 - text_h - 2 * pad
    txt_y2 = y1
    if txt_y1 < 0:
        # place inside box
        txt_y1 = y1 + pad
        txt_y2 = txt_y1 + text_h + pad
    txt_x2 = txt_x1 + text_w + 2 * pad
    # clamp horizontal
    if txt_x2 > w:
        shift = txt_x2 - w
        txt_x1 = max(0, txt_x1 - shift)
        txt_x2 = txt_x1 + text_w + 2 * pad
    # draw filled rectangle for text background
    cv2.rectangle(img, (txt_x1, txt_y1), (txt_x2, txt_y2), color, -1)
    # put text (white)
    text_org = (txt_x1 + pad, txt_y2 - pad - baseline)
    cv2.putText(img, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _box_iou(xyxy_a, xyxy_b):
    if not xyxy_a or not xyxy_b or len(xyxy_a) < 4 or len(xyxy_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, xyxy_a[:4])
    bx1, by1, bx2, by2 = map(float, xyxy_b[:4])
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _box_metrics(xyxy, img_w: int, img_h: int):
    if not xyxy or len(xyxy) < 4 or img_w <= 0 or img_h <= 0:
        return 0.0, 0.0, False
    x1, y1, x2, y2 = map(float, xyxy[:4])
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = bw * bh
    area_ratio = area / float(img_w * img_h)
    aspect = (max(bw, bh) / max(1e-6, min(bw, bh))) if bw > 0 and bh > 0 else 0.0
    near_full = (x1 <= 0.02 * img_w and y1 <= 0.02 * img_h and x2 >= 0.98 * img_w and y2 >= 0.98 * img_h)
    return area_ratio, aspect, near_full


def accept_custom_detection(det: dict, img_w: int, img_h: int, conf_thresh: float, min_area_pct: float = 0.1) -> bool:
    cls_name = str(det.get('class', '')).lower().strip()
    confv = float(det.get('conf', 0.0))
    if confv < conf_thresh:
        return False
    area_ratio, aspect, near_full = _box_metrics(det.get('xyxy'), img_w, img_h)
    min_area = max(0.0008, max(0.0, min_area_pct) / 100.0)

    if area_ratio <= 0:
        return False
    if near_full and confv < 0.60:
        return False

    if cls_name == 'flower':
        if area_ratio < min_area or area_ratio > 0.35:
            return False
        if aspect > 3.5:
            return False
    elif cls_name in {'fruit', 'apple', 'banana', 'orange'}:
        if area_ratio < min_area or area_ratio > 0.30:
            return False
        if aspect > 3.0:
            return False
    elif cls_name == 'dumbbell':
        if area_ratio < min_area or area_ratio > 0.40:
            return False
    elif cls_name == 'tree':
        if area_ratio < min_area:
            return False
        if area_ratio > 0.92 and confv < 0.55:
            return False

    return True


def refine_fruit_labels_with_coco(custom_dets: list, coco_dets: list):
    """Replace generic 'fruit' with concrete COCO fruit labels when boxes overlap."""
    # Only refine when COCO gives a very high-confidence fruit and strong overlap.
    coco_fruit_names = {'apple', 'banana', 'orange', 'watermelon'}
    coco_fruits = [d for d in (coco_dets or []) if str(d.get('class', '')).lower().strip() in coco_fruit_names]
    for d in custom_dets or []:
        cls_name = str(d.get('class', '')).lower().strip()
        if cls_name != 'fruit':
            continue
        best = None
        best_score = -1.0
        for c in coco_fruits:
            coco_conf = float(c.get('conf', 0.0))
            # require a very high-confidence COCO fruit and very strong overlap before mapping
            if coco_conf < 0.90:
                continue
            iou = _box_iou(d.get('xyxy'), c.get('xyxy'))
            if iou < 0.70:
                continue
            score = iou * coco_conf
            if score > best_score:
                best_score = score
                best = c
        if best is not None:
            d['class'] = str(best.get('class', 'fruit')).lower().strip()
            d['_refined_from'] = 'fruit'


def add_non_overlapping_detections(base_dets: list, extra_dets: list, iou_threshold: float = 0.5):
    for e in extra_dets or []:
        keep = True
        for b in base_dets:
            if str(b.get('class', '')).lower().strip() != str(e.get('class', '')).lower().strip():
                continue
            if _box_iou(b.get('xyxy'), e.get('xyxy')) >= iou_threshold:
                keep = False
                break
        if keep:
            base_dets.append(e)


def dedup_detections_by_class_nms(detections: list, iou_threshold: float = 0.45):
    grouped = {}
    for d in detections or []:
        cls_name = str(d.get('class', '')).lower().strip()
        grouped.setdefault(cls_name, []).append(d)

    out = []
    for cls_name, dets in grouped.items():
        ordered = sorted(dets, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
        kept = []
        for d in ordered:
            overlapped = False
            for k in kept:
                if _box_iou(d.get('xyxy'), k.get('xyxy')) >= iou_threshold:
                    overlapped = True
                    break
            if not overlapped:
                kept.append(d)
        out.extend(kept)
    return out


def dedup_detections_by_class_nms_classwise(detections: list, default_iou: float = 0.45):
    # Class-aware IoU thresholds to avoid undercount in dense flower scenes.
    # Higher IoU threshold => keep more overlapping same-class boxes.
    class_iou = {
        'flower': 0.92,
        'fruit': 0.65,
        'tree': 0.50,
        'dumbbell': 0.45,
        'unknown': 0.50,
    }
    grouped = {}
    for d in detections or []:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        d2 = dict(d)
        d2['class'] = cls_name
        grouped.setdefault(cls_name, []).append(d2)

    out = []
    for cls_name, dets in grouped.items():
        iou_threshold = float(class_iou.get(cls_name, default_iou))
        ordered = sorted(dets, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
        kept = []
        for d in ordered:
            overlapped = False
            for k in kept:
                if _box_iou(d.get('xyxy'), k.get('xyxy')) >= iou_threshold:
                    overlapped = True
                    break
            if not overlapped:
                kept.append(d)
        out.extend(kept)
    return out


def normalize_specialist_detections(detections: list, source: str):
    out = []
    for d in detections or []:
        d2 = dict(d)
        cls_name = str(d2.get('class', '')).lower().strip()
        if source == 'flower':
            d2['class'] = 'flower'
        elif source == 'dumbbell':
            d2['class'] = 'dumbbell'
        elif source == 'tree':
            d2['class'] = 'tree'
        elif source == 'fruit':
            d2['class'] = 'fruit'
        d2['_source_model'] = source
        out.append(d2)
    return out


def save_analysis(filename: str, objects: dict, media_type: str = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO analyses (filename, timestamp, media_type, objects_json) VALUES (?, ?, ?, ?)",
                (filename, datetime.utcnow().isoformat(), media_type, json.dumps(objects, ensure_ascii=False)))
    conn.commit()
    conn.close()


def list_history():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, filename, timestamp, media_type, objects_json FROM analyses ORDER BY id DESC LIMIT 200")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_description(name: str) -> str:
    mapping = {
        'person': 'A human being; often the primary subject in images.',
        'car': 'A vehicle for road transport.',
        'bicycle': 'A two-wheeled vehicle powered by pedaling.',
        'dog': 'A domestic animal often kept as a pet.',
        'cat': 'A small domesticated feline animal.',
        'chair': 'A piece of furniture for sitting.',
        'bottle': 'A container for liquids.'
    }
    # Accept either English or Vietnamese labels: map Vietnamese back to English when possible
    key = name.lower()
    if key in mapping:
        return mapping[key]
    # reverse lookup: check if name is Vietnamese label
    for en, vn in VN_LABELS.items():
        if vn == name or vn == name.lower():
            return mapping.get(en, f'A detected object of type "{name}".')
    return mapping.get(key, f'A detected object of type "{name}".')


def fetch_wikipedia_summary(name: str, lang: str = 'en') -> dict:
    """Fetch a short summary from Wikipedia REST API. Returns dict with title, extract, url.
    Caches results in session_state under `wiki_cache` to avoid repeated requests.
    """
    if not name:
        return {}
    cache = st.session_state.get('wiki_cache', {})
    key = f"{lang}:{name.lower()}"
    if key in cache:
        return cache[key]
    try:
        title = name.replace(' ', '_')
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            out = {
                'title': data.get('title'),
                'extract': data.get('extract'),
                'url': data.get('content_urls', {}).get('desktop', {}).get('page') if isinstance(data.get('content_urls', {}), dict) else None
            }
        else:
            out = {}
    except Exception:
        out = {}
    cache[key] = out
    st.session_state['wiki_cache'] = cache
    return out


def main():
    st.set_page_config(
        page_title="SmartFocus AI - Dashboard",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: #0b1220;
            color: #e5e7eb;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #0b1220;
        }
        .main-header {
            font-size: 1.75rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.2rem;
            letter-spacing: -0.025em;
        }
        .main-sub {
            color: #94a3b8;
            font-size: 0.92rem;
            margin-bottom: 0.2rem;
        }
        .sf-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.35);
            margin-bottom: 14px;
        }
        div[data-testid="stMetric"] {
            background-color: #111827;
            border: 1px solid #1f2937;
            padding: 1.25rem;
            border-radius: 12px;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.35);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #e5e7eb !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1f2937;
        }
        section[data-testid="stSidebar"] * {
            color: #dbeafe;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #334155;
            background-color: #0f172a;
            color: #e2e8f0;
            font-weight: 500;
            transition: all 0.2s;
            min-height: 40px;
        }
        .stButton>button:hover {
            border-color: #60a5fa;
            color: #f8fafc;
            background-color: #1e293b;
        }
        div[data-testid="stFileUploader"] {
            border: 1px dashed #334155;
            border-radius: 12px;
            background: #0f172a;
            padding: 8px;
        }
        div[data-testid="stFileUploader"] * {
            color: #dbeafe !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #1f2937;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.35);
            background: #0f172a;
        }
        div[data-testid="stDataFrame"] * {
            color: #e5e7eb !important;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stCaption, .stText {
            color: #d1d5db;
        }
        hr {
            border-color: #1f2937 !important;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            max-width: 1400px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    init_db()
    if 'last_total_objects' not in st.session_state:
        st.session_state.last_total_objects = 0
    if 'last_accuracy' not in st.session_state:
        st.session_state.last_accuracy = 0.0
    if 'last_latency_ms' not in st.session_state:
        st.session_state.last_latency_ms = 0
    if 'last_load_pct' not in st.session_state:
        st.session_state.last_load_pct = 0.0
    if 'last_top_label' not in st.session_state:
        st.session_state.last_top_label = ""

    col_nav_1, col_nav_2 = st.columns([1, 1])
    with col_nav_1:
        st.markdown("<h1 class='main-header'>🎯 SmartFocus AI</h1>", unsafe_allow_html=True)
        st.markdown("<div class='main-sub'>Nền tảng nhận diện vật thể thời gian thực</div>", unsafe_allow_html=True)
    with col_nav_2:
        st.markdown(
            "<div style='text-align: right; color: #6b7280; font-size: 0.875rem; padding-top: 10px;'>Hệ thống trực tuyến • v2.0.4</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    with st.sidebar:
        st.markdown("### Cấu hình hệ thống")
        st.write("Thiết lập tham số cho mô hình nhận diện.")

        model_type = st.selectbox(
            "Kiến trúc mô hình",
            ["Kết hợp (COCO + Custom4)", "Custom4", "YOLOv11s (Small)", "YOLOv11n (Nano)"],
            index=0,
        )

        confidence = st.slider(
            "Ngưỡng tin cậy (Confidence)",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
        )

        imgsz = st.select_slider("Image size", options=IMG_SIZE_OPTIONS, value=640)

        enable_dense = st.checkbox(
            "Dense Flower Mode (tiled fallback)",
            value=ENABLE_DENSE_FLOWER_FALLBACK,
            help="Enable tiled inference fallback for dense flower scenes to improve counting.",
        )
        st.session_state['enable_dense_flower_fallback'] = bool(enable_dense)

        st.markdown("---")
        st.markdown("### Thông tin Node")
        st.info("Trạng thái: Đang hoạt động\n\nNode ID: SF-VN-01")

        if st.button("Làm mới bộ nhớ đệm"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            st.toast("Đã xóa cache hệ thống!")

    st.markdown("### ✅ SMARTFOCUS UI VERSION 2.0 - STREAMLIT")

    st.markdown("<div class='sf-card'>", unsafe_allow_html=True)
    st.markdown("#### Phân tích dữ liệu mới")
    uploaded_file = st.file_uploader("Kéo và thả tệp tin vào đây (Ảnh hoặc Video)", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    st.markdown("</div>", unsafe_allow_html=True)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    metric_obj = m_col1.empty()
    metric_acc = m_col2.empty()
    metric_lat = m_col3.empty()
    metric_load = m_col4.empty()

    def render_metrics(total: int, acc_pct: float, latency: int, load_pct: float, top_label_text: str):
        metric_obj.metric(
            label="Vật thể phát hiện",
            value=str(max(0, int(total))),
            delta=(top_label_text or "—"),
            delta_color="off",
        )
        metric_acc.metric(label="Độ chính xác trung bình", value=f"{float(acc_pct):.1f}%", delta="")
        metric_lat.metric(label="Độ trễ xử lý", value=f"{int(latency)} ms", delta="", delta_color="normal")
        metric_load.metric(label="Tải lượng hệ thống", value=f"{float(load_pct):.1f}%", delta="")

    render_metrics(
        int(st.session_state.last_total_objects),
        float(st.session_state.last_accuracy),
        int(st.session_state.last_latency_ms),
        float(st.session_state.last_load_pct),
        str(st.session_state.last_top_label or ""),
    )

    if uploaded_file is not None:
        is_image = (uploaded_file.type or "").startswith("image")
        file_bytes = uploaded_file.read()

        model_path_map = {
            "YOLOv11n (Nano)": "yolo11n.pt",
            "YOLOv11s (Small)": "yolo11s.pt",
            "Custom4": CUSTOM_MODEL_PATH,
        }

        t0 = datetime.now()
        with st.spinner('Đang xử lý dữ liệu...'):
            try:
                if model_type == "Kết hợp (COCO + Custom4)":
                    coco_model = load_model(COCO_MODEL_PATH)
                    ann_coco, counts_coco, raw_coco, raw_img = process_uploaded(
                        coco_model,
                        file_bytes,
                        is_image,
                        float(confidence),
                        imgsz=int(imgsz),
                        use_fp16=(torch is not None and torch.cuda.is_available()),
                        max_frames=60,
                        sample_rate=3,
                        batch_size=4,
                        nms_iou=float(st.session_state.get('nms_iou', 0.5)),
                        max_det=int(st.session_state.get('max_det', 120)),
                    )

                    if os.path.exists(CUSTOM_MODEL_PATH):
                        custom_model = load_model(CUSTOM_MODEL_PATH)
                        ann_custom, counts_custom, raw_custom, _ = process_uploaded(
                            custom_model,
                            file_bytes,
                            is_image,
                            float(max(0.15, confidence)),
                            imgsz=int(imgsz),
                            use_fp16=(torch is not None and torch.cuda.is_available()),
                            max_frames=60,
                            sample_rate=3,
                            batch_size=4,
                            nms_iou=float(st.session_state.get('nms_iou', 0.5)),
                            max_det=int(st.session_state.get('max_det', 120)),
                        )
                        merged_dets = []
                        if isinstance(raw_coco, dict):
                            merged_dets.extend(raw_coco.get('detections', []) or [])
                        if isinstance(raw_custom, dict):
                            merged_dets.extend(raw_custom.get('detections', []) or [])

                        # Post-process merged detections to reduce obvious hybrid errors
                        cls_alias = {}
                        cleaned = []
                        img_h, img_w = (raw_img.shape[0], raw_img.shape[1]) if (is_image and raw_img is not None) else (0, 0)
                        for d in merged_dets:
                            cls_name = str(d.get('class', '')).lower().strip()
                            cls_name = cls_alias.get(cls_name, cls_name)
                            d['class'] = cls_name
                            confv = float(d.get('conf', 0.0))
                            if confv < max(0.12, float(confidence) - 0.08):
                                continue

                            if img_w > 0 and img_h > 0:
                                area_ratio, aspect, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)
                                if area_ratio <= 0:
                                    continue
                                # keep tiny boxes only when confidence is reasonably high
                                if area_ratio < 0.0006 and confv < 0.45:
                                    continue
                                if near_full and confv < 0.85:
                                    continue
                                # do not aggressively drop large flowers (single-object photos)
                                if cls_name == 'flower' and area_ratio > 0.92 and confv < 0.80:
                                    continue
                                if cls_name == 'fruit' and area_ratio > 0.65 and confv < 0.85:
                                    continue
                                if cls_name == 'tree' and area_ratio > 0.90 and confv < 0.90:
                                    continue

                            cleaned.append(d)

                        # If custom flower exists, suppress non-custom class boxes overlapping flower boxes
                        custom_classes = {'flower', 'fruit', 'tree', 'dumbbell'}
                        has_custom_flower = any(
                            str(d.get('class', '')).lower().strip() == 'flower'
                            for d in (raw_custom.get('detections', []) if isinstance(raw_custom, dict) else [])
                        )
                        if has_custom_flower:
                            pref_boxes = [d for d in cleaned if str(d.get('class', '')).lower().strip() in custom_classes]
                            reduced = []
                            for d in cleaned:
                                cls_name = str(d.get('class', '')).lower().strip()
                                if cls_name not in custom_classes:
                                    overlap_pref = any(_box_iou(d.get('xyxy'), p.get('xyxy')) >= 0.45 for p in pref_boxes)
                                    if overlap_pref:
                                        continue
                                reduced.append(d)
                            cleaned = reduced

                        merged_dets = dedup_detections_by_class_nms_classwise(cleaned, default_iou=0.45)

                        # Specialist verification for custom classes to reduce false positives
                        if is_image and raw_img is not None:
                            custom_raw_dets = (raw_custom.get('detections', []) if isinstance(raw_custom, dict) else [])
                            refined_custom = refine_custom_detections_with_specialists(
                                raw_img,
                                custom_raw_dets,
                                base_conf=float(confidence),
                                imgsz=int(imgsz),
                                use_fp16=(torch is not None and torch.cuda.is_available()),
                            )
                            # replace custom-class boxes in merged set with refined specialist-backed boxes
                            non_custom = [
                                d for d in merged_dets
                                if normalize_class_name(d.get('class', '')) not in {'flower', 'fruit', 'tree', 'dumbbell'}
                            ]
                            merged_dets = non_custom + refined_custom
                            merged_dets = dedup_detections_by_class_nms_classwise(merged_dets, default_iou=0.50)

                        # root-fix: if strict hybrid post-filtering removes all objects,
                        # fallback to a softer merge so valid detections are not lost.
                        if not merged_dets:
                            soft = []
                            for d in (raw_custom.get('detections', []) if isinstance(raw_custom, dict) else []):
                                d2 = dict(d)
                                c2 = str(d2.get('class', '')).lower().strip()
                                d2['class'] = cls_alias.get(c2, c2) or 'unknown'
                                conf2 = float(d2.get('conf', 0.0))
                                if img_w > 0 and img_h > 0:
                                    area_ratio2, _, near_full2 = _box_metrics(d2.get('xyxy'), img_w, img_h)
                                    if area_ratio2 <= 0:
                                        continue
                                    if d2['class'] == 'flower' and area_ratio2 > 0.92 and conf2 < 0.85:
                                        continue
                                    if near_full2 and conf2 < 0.90:
                                        continue
                                if conf2 >= max(0.08, float(confidence) * 0.45):
                                    soft.append(d2)
                            for d in (raw_coco.get('detections', []) if isinstance(raw_coco, dict) else []):
                                d2 = dict(d)
                                c2 = str(d2.get('class', '')).lower().strip()
                                d2['class'] = cls_alias.get(c2, c2) or 'unknown'
                                conf2 = float(d2.get('conf', 0.0))
                                if img_w > 0 and img_h > 0:
                                    area_ratio2, _, near_full2 = _box_metrics(d2.get('xyxy'), img_w, img_h)
                                    if area_ratio2 <= 0:
                                        continue
                                    if d2['class'] == 'flower' and area_ratio2 > 0.92 and conf2 < 0.85:
                                        continue
                                    if near_full2 and conf2 < 0.92:
                                        continue
                                if conf2 >= max(0.12, float(confidence) * 0.55):
                                    soft.append(d2)
                            merged_dets = dedup_detections_by_class_nms_classwise(soft, default_iou=0.50)

                        # Final verified filter for all classes to reduce over-counting (especially COCO noise)
                        pre_verified_merged = dedup_detections_by_class_nms_classwise(merged_dets, default_iou=0.50)
                        merged_dets = verify_and_reduce_detections(
                            merged_dets,
                            img_w=img_w,
                            img_h=img_h,
                            base_conf=float(confidence),
                            strict_coco=True,
                        )

                        # Root-fix: keep non-empty pre-verified result if strict verifier becomes over-aggressive.
                        if not merged_dets and pre_verified_merged:
                            merged_dets = pre_verified_merged

                        if is_image and raw_img is not None:
                            ann = overlay_detections(raw_img, merged_dets, conf_thresh=float(max(0.15, confidence)))
                        else:
                            ann = ann_coco if ann_coco is not None else ann_custom

                        final_detections = canonicalize_final_detections(merged_dets)
                        counts = build_counts_from_detections(final_detections)
                        raw = {"detections": final_detections, "mode": "hybrid"}
                    else:
                        st.warning(f"Không tìm thấy {CUSTOM_MODEL_PATH}, đang fallback sang COCO.")
                        ann, counts, raw = ann_coco, counts_coco, raw_coco
                else:
                    selected_model = model_path_map.get(model_type, "yolo11s.pt")
                    model = load_model(selected_model)
                    ann, counts, raw, raw_img = process_uploaded(
                        model,
                        file_bytes,
                        is_image,
                        float(confidence),
                        imgsz=int(imgsz),
                        use_fp16=(torch is not None and torch.cuda.is_available()),
                        max_frames=60,
                        sample_rate=3,
                        batch_size=4,
                        nms_iou=float(st.session_state.get('nms_iou', 0.5)),
                        max_det=int(st.session_state.get('max_det', 120)),
                    )
                    if is_image and model_type == "Custom4" and raw_img is not None:
                        refined_custom = refine_custom_detections_with_specialists(
                            raw_img,
                            (raw.get('detections', []) if isinstance(raw, dict) else []),
                            base_conf=float(confidence),
                            imgsz=int(imgsz),
                            use_fp16=(torch is not None and torch.cuda.is_available()),
                        )
                        ann = overlay_detections(raw_img, refined_custom, conf_thresh=float(max(0.15, confidence)))
                        counts = build_counts_from_detections(refined_custom)
                        raw = {"detections": canonicalize_final_detections(refined_custom), "mode": "custom_refined"}
            except Exception as e:
                st.error(f"Lỗi xử lý: {e}")
                ann, counts, raw = None, {}, {"detections": []}

        latency_ms = int((datetime.now() - t0).total_seconds() * 1000)
        final_detections = extract_final_detections_from_raw(raw)

        # Single source of truth: always rebuild counts from final_detections
        rebuilt_counts = build_counts_from_detections(final_detections)
        counts = rebuilt_counts if rebuilt_counts else (counts if isinstance(counts, dict) else {})

        # If we only had frames before, persist canonical detections for summary/history
        if isinstance(raw, dict):
            raw['detections'] = final_detections

        total_objects = len(final_detections)
        avg_conf = float(sum(float(d.get('conf', 0.0)) for d in final_detections) / len(final_detections)) if final_detections else 0.0

        st.session_state.last_total_objects = total_objects
        st.session_state.last_accuracy = avg_conf * 100.0
        st.session_state.last_latency_ms = latency_ms
        st.session_state.last_load_pct = min(99.0, max(1.0, total_objects * 2.5))
        # compute top detected class name (english) and store its Vietnamese translation for UI
        try:
            if counts:
                top_class = max(counts.items(), key=lambda x: x[1])[0]
                st.session_state.last_top_label = translate_label(str(top_class))
            elif final_detections:
                cls_from_det = str(final_detections[0].get('class', '')).strip()
                st.session_state.last_top_label = translate_label(cls_from_det) if cls_from_det else 'Unknown'
            else:
                st.session_state.last_top_label = ""
        except Exception:
            st.session_state.last_top_label = ""

        render_metrics(
            int(st.session_state.last_total_objects),
            float(st.session_state.last_accuracy),
            int(st.session_state.last_latency_ms),
            float(st.session_state.last_load_pct),
            str(st.session_state.last_top_label or ""),
        )

        st.success(f"Đã phân tích xong tệp: {uploaded_file.name}")

        c1, c2 = st.columns([2.2, 1], gap="large")
        with c1:
            st.markdown("<div class='sf-card'>", unsafe_allow_html=True)
            if ann is not None:
                st.image(ann, channels="RGB", caption="Kết quả nhận diện thực tế", use_container_width=True)
            else:
                st.info("Chưa có ảnh kết quả để hiển thị.")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='sf-card'>", unsafe_allow_html=True)
            st.markdown("##### Chi tiết phát hiện")
            top_items = sorted((counts or {}).items(), key=lambda x: x[1], reverse=True)[:8]
            st.json({
                "tong_vat_the": total_objects,
                "confidence_tb": round(avg_conf, 3),
                "thoi_gian_xu_ly_ms": latency_ms,
                "top_lop": [{"lop": translate_label(k), "so_luong": int(v)} for k, v in top_items],
            })
            st.markdown("</div>", unsafe_allow_html=True)

        save_analysis(uploaded_file.name, {"counts": counts, "raw": raw}, media_type=('image' if is_image else 'video'))

    st.markdown("#### Nhật ký hệ thống")
    rows = list_history()
    history_data = []
    for r in rows[:100]:
        _id, fname, ts, media_type, objs = r
        try:
            parsed = json.loads(objs)
        except Exception:
            parsed = {}
        counts_h = parsed.get('counts', {}) if isinstance(parsed, dict) else {}
        total_h = int(sum(counts_h.values())) if isinstance(counts_h, dict) else 0
        dets_h = parsed.get('raw', {}).get('detections', []) if isinstance(parsed, dict) and isinstance(parsed.get('raw', {}), dict) else []
        conf_h = (sum(float(d.get('conf', 0.0)) for d in dets_h) / len(dets_h) * 100.0) if dets_h else 0.0
        history_data.append({
            "Thời gian": ts,
            "Tên tệp": fname,
            "Loại": "Ảnh" if media_type == 'image' else "Video",
            "Số vật thể": total_h,
            "Độ chính xác": f"{conf_h:.1f}%",
        })

    if history_data:
        st.dataframe(history_data, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có dữ liệu lịch sử.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #9ca3af; font-size: 0.75rem;'>"
        "Thiết kế bởi SmartFocus AI Team © 2024. Bảo lưu mọi quyền."
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
