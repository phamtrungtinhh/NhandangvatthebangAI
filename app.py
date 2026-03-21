import streamlit as st
import sys
import os
import glob
import sqlite3
import json
import streamlit as st
import sqlite3
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import requests
from pathlib import Path

from PIL import Image
import io
import numpy as np
import cv2
import unicodedata
import math
import startup

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = str(PROJECT_ROOT / "analysis.db")
IMG_SIZE_OPTIONS = [320, 416, 512, 640, 768, 1024]
CUSTOM_MODEL_PATH = str(PROJECT_ROOT / "model_custom" / "weights" / "custo_all.pt")
FLOWER_SPECIALIST_MODEL_PATH = str(PROJECT_ROOT / "models" / "flower_best.pt")
DUMBBELL_SPECIALIST_MODEL_PATH = str(PROJECT_ROOT / "models" / "dumbbell_best.pt")
FRUIT_SPECIALIST_MODEL_PATH = str(PROJECT_ROOT / "models" / "fruit_best.pt")
TREE_SPECIALIST_MODEL_PATH = str(PROJECT_ROOT / "models" / "tree_best.pt")
USE_SPECIALIST_MODELS = False
HYBRID_MODEL_TYPE = "Kết hợp (COCO + Custom4)"
CUSTOM_ONLY_MODEL_TYPE = "Custom4"
COCO_SMALL_MODEL_TYPE = "YOLOv11s (Small)"
COCO_NANO_MODEL_TYPE = "YOLOv11n (Nano)"
DEFAULT_MODEL_TYPE = CUSTOM_ONLY_MODEL_TYPE
HYBRID_STRICT_SOURCE_ISOLATION = True
HYBRID_PARALLEL_INFERENCE = True
CUSTOM_MODEL_METADATA_PATH = str(PROJECT_ROOT / "model_custom" / "weights" / "custo_all.deployment.json")
FLOWER_MODEL_PATH = CUSTOM_MODEL_PATH
COCO_MODEL_PATH = str(PROJECT_ROOT / "yolo11n.pt")
CUSTOM_CLASSES = {"flower", "fruit", "tree", "dumbbell"}
ENABLE_DENSE_FLOWER_FALLBACK = False
MIN_COUNT_CONFIDENCE = 0.30
APP_BUILD_ID = "2026-03-07-threshold-regression-fix"
SHOW_DETAIL_PANEL = bool(int(os.environ.get("SHOW_DETAIL_PANEL", "0")))
FAST_TILE_SIZE = 640
FAST_TILE_STRIDE = 320
FAST_TILE_BATCH_SIZE = 8
IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_FILE_EXTENSIONS = {".mp4", ".avi", ".mov"}
SUPPORTED_MEDIA_EXTENSIONS = IMAGE_FILE_EXTENSIONS | VIDEO_FILE_EXTENSIONS
SPECIFIC_FRUIT_CLASSES = {
    'apple',
    'banana',
    'orange',
    'watermelon',
    'mango',
    'strawberry',
    'grape',
    'lemon',
    'lime',
    'pear',
    'peach',
    'plum',
    'cherry',
    'pineapple',
    'avocado',
    'kiwi',
}
CROSS_CLASS_SUPPRESS_SET = {
    'flower',
    'fruit',
    'tree',
    'dumbbell',
    'vase',
    'apple',
    'banana',
    'orange',
    'watermelon',
    'mango',
    'strawberry',
    'grape',
    'lemon',
    'lime',
    'pear',
    'peach',
    'plum',
    'cherry',
    'pineapple',
    'avocado',
    'kiwi',
}
# Environment-overridable gates for targeted classes (useful for focused retests)
# Dumbbell: minimum confidence and minimum area ratio to keep detection
DUMBBELL_MIN_CONF = float(os.environ.get("FLOWER_GATE_DUMBBELL_CONF", "0.20"))
DUMBBELL_MIN_AREA = float(os.environ.get("FLOWER_GATE_DUMBBELL_AREA", "0.00005"))
# Tree: minimum confidence and minimum area ratio to keep detection
TREE_MIN_CONF = float(os.environ.get("FLOWER_GATE_TREE_CONF", "0.18"))
TREE_MIN_AREA = float(os.environ.get("FLOWER_GATE_TREE_AREA", "0.00006"))
KEEP_COCO_OVERLAPS_WITH_CUSTOM = True
_CUSTOM_CLASS_CACHE = None
_FACE_CASCADE = None

MODEL_TYPE_ALIASES = {
    HYBRID_MODEL_TYPE: HYBRID_MODEL_TYPE,
    "Káº¿t há»£p (COCO + Custom4)": HYBRID_MODEL_TYPE,
    "Ket hop (COCO + Custom4)": HYBRID_MODEL_TYPE,
    CUSTOM_ONLY_MODEL_TYPE: CUSTOM_ONLY_MODEL_TYPE,
    "Tùy chỉnh4": CUSTOM_ONLY_MODEL_TYPE,
    "Tuỳ chỉnh4": CUSTOM_ONLY_MODEL_TYPE,
    "Tuy chinh4": CUSTOM_ONLY_MODEL_TYPE,
    COCO_SMALL_MODEL_TYPE: COCO_SMALL_MODEL_TYPE,
    "YOLOv11s (Nhỏ)": COCO_SMALL_MODEL_TYPE,
    COCO_NANO_MODEL_TYPE: COCO_NANO_MODEL_TYPE,
}

# Confusion pairs for source-aware suppression (only handle sensitive pairs)
CONFUSION_PAIRS = {
    frozenset(('flower', 'fruit')),
    frozenset(('flower', 'tree')),
    frozenset(('flower', 'dumbbell')),
    frozenset(('flower', 'vase')),
    frozenset(('orange', 'banana')),
    frozenset(('orange', 'fruit')),
    frozenset(('banana', 'fruit')),
}

# Quick source/class priority maps (can be tuned later)
SOURCE_PRIORITY = {
    'custom': 0.15,
    'coco': 0.00,
}

CLASS_PRIORITY = {
    'flower': 0.08,
    'fruit': 0.04,
    'tree': 0.04,
    'dumbbell': 0.04,
    'vase': 0.04,
    'orange': 0.10,
    'banana': 0.09,
}


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


def get_model_cache_token(model_name: str):
    try:
        if os.path.exists(model_name):
            stat = os.stat(model_name)
            return (os.path.abspath(model_name), int(stat.st_size), int(stat.st_mtime_ns))
    except Exception:
        pass
    return (str(model_name), None, None)


@st.cache_resource
def _load_model_cached(model_name: str = "yolo11n.pt", cache_token=None):
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


def load_model(model_name: str = "yolo11n.pt"):
    return _load_model_cached(model_name, cache_token=get_model_cache_token(model_name))


def load_custom_model():
    custom_model_path = CUSTOM_MODEL_PATH
    return _load_model_cached(
        custom_model_path,
        cache_token=get_model_cache_token(custom_model_path),
    )


@st.cache_resource
def load_default_detection_models(coco_cache_token=None, custom_cache_token=None):
    coco_model = load_model(COCO_MODEL_PATH)
    custom_model = load_custom_model() if os.path.exists(CUSTOM_MODEL_PATH) else None
    return coco_model, custom_model


@st.cache_data(show_spinner=False)
def read_model_metadata(metadata_path: str = CUSTOM_MODEL_METADATA_PATH) -> dict:
    try:
        metadata_file = Path(metadata_path)
        if metadata_file.exists():
            return json.loads(metadata_file.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def check_class_order_and_warn(
    data_yaml_path: str = str(PROJECT_ROOT / "model_custom" / "dataset" / "data.yaml"),
    deployment_glob: str = str(PROJECT_ROOT / "model_custom" / "weights" / "*.deployment.json"),
) -> dict:
    """Compare dataset class order vs deployment metadata and log a warning if mismatch.

    Returns a dict with keys: dataset_names, deployment_names (first found), mismatch (bool).
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    dataset_names = []
    deployment_names = []
    try:
        if os.path.exists(data_yaml_path):
            try:
                import yaml
                with open(data_yaml_path, 'r', encoding='utf-8') as fh:
                    y = yaml.safe_load(fh)
                    # data.yaml can have 'names' as list or dict
                    names = y.get('names') if isinstance(y, dict) else None
                    if isinstance(names, dict):
                        # dict mapping idx->name: convert to ordered list by key
                        dataset_names = [v for k, v in sorted(names.items(), key=lambda x: int(x[0]))]
                    elif isinstance(names, list):
                        dataset_names = names
            except Exception:
                pass
    except Exception:
        pass

    # find first deployment json
    try:
        for fn in sorted(glob.glob(deployment_glob)):
            try:
                j = json.loads(Path(fn).read_text(encoding='utf-8'))
                cls = j.get('class_names') or j.get('names') or {}
                if isinstance(cls, dict):
                    deployment_names = [v for k, v in sorted(cls.items(), key=lambda x: int(x[0]))]
                elif isinstance(cls, list):
                    deployment_names = cls
                if deployment_names:
                    break
            except Exception:
                continue
    except Exception:
        pass

    mismatch = False
    try:
        if dataset_names and deployment_names:
            if len(dataset_names) != len(deployment_names) or any(a != b for a, b in zip(dataset_names, deployment_names)):
                mismatch = True
                logging.warning(f"Class order/length mismatch between {data_yaml_path} and deployment metadata {deployment_glob}.")
                logging.warning(f"Dataset names: {dataset_names}")
                logging.warning(f"Deployment names: {deployment_names}")
    except Exception:
        pass

    return {'dataset_names': dataset_names, 'deployment_names': deployment_names, 'mismatch': mismatch}


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

        # Low-count rescue for custom tree/fruit to reduce under-counting.
        try:
            if prefer_recall_custom:
                custom_classes = get_custom_class_names()
                tree_n = int(
                    sum(
                        1
                        for d in detections
                        if (normalize_class_name(d.get('class', '')) or '') == 'tree'
                    )
                )
                if 'tree' in custom_classes and tree_n <= 1:
                    rescue_conf = max(0.10, float(tuned_conf) * 0.45)
                    rescue_iou = min(0.60, float(tuned_nms_iou) + 0.05)
                    rescue_max_det = max(200, int(tuned_max_det))
                    rescue_imgsz = int(max(int(imgsz), 768))
                    _, _, rescue_dets = run_detection_on_image(
                        model,
                        img,
                        rescue_conf,
                        imgsz=rescue_imgsz,
                        use_fp16=use_fp16,
                        nms_iou=rescue_iou,
                        max_det=rescue_max_det,
                    )
                    rescue_dets = [
                        d for d in canonicalize_final_detections(rescue_dets)
                        if (normalize_class_name(d.get('class', '')) or '') == 'tree'
                        and float(d.get('conf', 0.0)) >= 0.10
                    ]
                    if rescue_dets:
                        detections = dedup_detections_by_class_nms_classwise(
                            detections + rescue_dets,
                            default_iou=0.60,
                        )
                        stage_stats['after_tree_lowcount_rescue'] = summarize_detection_stage(detections)

                fruit_n = int(
                    sum(
                        1
                        for d in detections
                        if (normalize_class_name(d.get('class', '')) or '') in ({'fruit'} | set(SPECIFIC_FRUIT_CLASSES))
                    )
                )
                if 'fruit' in custom_classes and fruit_n == 0:
                    rescue_conf = max(0.10, float(tuned_conf) * 0.45)
                    rescue_iou = min(0.60, float(tuned_nms_iou) + 0.05)
                    rescue_max_det = max(200, int(tuned_max_det))
                    rescue_imgsz = int(max(int(imgsz), 768))
                    _, _, rescue_dets = run_detection_on_image(
                        model,
                        img,
                        rescue_conf,
                        imgsz=rescue_imgsz,
                        use_fp16=use_fp16,
                        nms_iou=rescue_iou,
                        max_det=rescue_max_det,
                    )
                    rescue_dets = [
                        d for d in canonicalize_final_detections(rescue_dets)
                        if (normalize_class_name(d.get('class', '')) or '') == 'fruit'
                        and float(d.get('conf', 0.0)) >= 0.14
                    ]
                    if rescue_dets:
                        detections = dedup_detections_by_class_nms_classwise(
                            detections + rescue_dets,
                            default_iou=0.60,
                        )
                        stage_stats['after_fruit_lowcount_rescue'] = summarize_detection_stage(detections)
        except Exception:
            pass
    # snap to nearest allowed size
    imgsz = min(IMG_SIZE_OPTIONS, key=lambda x: abs(x - imgsz))
    return conf, imgsz


def list_local_media_files(folder_path: str, recursive: bool = True):
    folder = Path(folder_path).expanduser()
    if not folder_path or not folder.exists():
        return [], "Không tìm thấy thư mục đã nhập."
    if not folder.is_dir():
        return [], "Đường dẫn đã nhập không phải là thư mục."

    try:
        pattern = "**/*" if recursive else "*"
        media_files = [
            path for path in folder.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
        ]
    except Exception as exc:
        return [], f"Không thể quét thư mục: {exc}"

    media_files.sort(key=lambda path: str(path).lower())
    return media_files, None


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


def auto_tune_image_inference_params(image_np: np.ndarray, base_conf: float, base_nms_iou: float = 0.5, base_max_det: int = 100):
    """Auto tune inference params from image complexity to stabilize counting accuracy.
    Returns: tuned_conf, tuned_nms_iou, tuned_max_det
    """
    # Counting-focused baseline: do not drop below strict confidence floor.
    conf = max(0.25, float(base_conf))
    nms_iou = float(base_nms_iou)
    max_det = int(base_max_det)

    if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
        return max(0.20, min(0.60, conf)), max(0.45, min(0.80, nms_iou)), max(50, min(260, max_det))

    h, w = image_np.shape[:2]
    if h <= 0 or w <= 0:
        return max(0.20, min(0.60, conf)), max(0.45, min(0.80, nms_iou)), max(50, min(260, max_det))

    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    except Exception:
        gray = np.mean(image_np, axis=2).astype(np.uint8) if image_np.ndim == 3 else image_np

    total_px = float(max(1, h * w))
    edge_density = 0.0
    blur_var = 0.0
    brightness = 0.5
    try:
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.count_nonzero(edges)) / total_px
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray)) / 255.0
    except Exception:
        pass

    megapixels = total_px / 1_000_000.0

    # Dense / textured scene: keep recall and make NMS more permissive so
    # neighboring overlapping flowers are not suppressed too early.
    if edge_density >= 0.11 or blur_var >= 500.0:
        conf -= 0.01
        nms_iou += 0.12
        max_det = max(max_det, 170)
    elif edge_density <= 0.035 and blur_var <= 120.0:
        # Sparse scene: reduce duplicate noise a bit.
        conf += 0.02
        nms_iou -= 0.03
        max_det = max(max_det, 90)

    # Lighting compensation
    if brightness <= 0.22:
        conf -= 0.03
    elif brightness >= 0.82:
        conf += 0.02

    # Larger images can contain more instances.
    if megapixels >= 1.6:
        max_det = max(max_det, 130)
    elif megapixels <= 0.4:
        max_det = max(max_det, 80)

    conf = max(0.20, min(0.60, conf))
    nms_iou = max(0.45, min(0.80, nms_iou))
    max_det = max(50, min(260, int(max_det)))
    return conf, nms_iou, max_det


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
    'horse': 'ngá»±a',
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
    'mouse': 'chuá»™t',
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
        'lo hoa': 'vase',
        'lọ hoa': 'vase',
        'vase': 'vase',
        'trai cay': 'fruit',
        'trái cây': 'fruit',
        'hoa qua': 'fruit',
        'fruit': 'fruit',
        'cam': 'orange',
        'orange': 'orange',
        'chuoi': 'banana',
        'chuoi tieu': 'banana',
        'banana': 'banana',
        'ta tay': 'dumbbell',
        'tạ tay': 'dumbbell',
        'dumbbell': 'dumbbell',
    }
    return alias.get(key, key)


# Expand to include specific fruit classes (flower vs apple/banana/...) to prefer specific fruits
for _fruit in SPECIFIC_FRUIT_CLASSES:
    try:
        CONFUSION_PAIRS.add(frozenset(('flower', normalize_class_name(_fruit))))
    except Exception:
        pass

# Prefer fruit-like classes over dumbbell when overlapping.
for _fruit in SPECIFIC_FRUIT_CLASSES:
    try:
        CONFUSION_PAIRS.add(frozenset(('dumbbell', normalize_class_name(_fruit))))
    except Exception:
        pass
CONFUSION_PAIRS.add(frozenset(('dumbbell', 'fruit')))


def get_custom_class_names() -> set:
    """Return the fixed custom classes supported by the Custom4 model.

    Important:
    - Keep this fixed to avoid accidental class leakage from stale deployment metadata.
    - Custom4 must only expose these classes in UI/inference: flower, fruit, tree, dumbbell.
    """
    global _CUSTOM_CLASS_CACHE
    if _CUSTOM_CLASS_CACHE is not None:
        return set(_CUSTOM_CLASS_CACHE)
    classes = set(CUSTOM_CLASSES)
    _CUSTOM_CLASS_CACHE = sorted(classes)
    return set(classes)


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
        entry = {
            'class': cls_name,
            'conf': confv,
            'xyxy': list(xyxy[:4]) if xyxy and len(xyxy) >= 4 else xyxy,
        }
        # preserve optional metadata like source model to allow source-aware rules
        try:
            for k, v in (d or {}).items():
                if k and isinstance(k, str) and k.startswith('_'):
                    entry[k] = v
        except Exception:
            pass
        out.append(entry)
    return out


def ensure_source_model(detections: list, source: str) -> list:
    """Ensure each detection has _source_model field; assign if missing.
    
    Args:
        detections: List of detection dicts
        source: "coco" or "custom"
    
    Returns:
        List of detections with _source_model guaranteed
    """
    dets = canonicalize_final_detections(detections)
    for d in dets:
        if '_source_model' not in d or not d.get('_source_model'):
            d['_source_model'] = source
    return dets


def normalize_ui_model_type(model_type: str) -> str:
    raw = str(model_type or "").strip()
    return MODEL_TYPE_ALIASES.get(raw, raw if raw else DEFAULT_MODEL_TYPE)


def filter_detections_by_mode(detections: list, mode: str) -> list:
    """Filter detections based on mode isolation rules.
    
    Args:
        detections: List of detection dicts
        mode: "coco" | "custom" | "hybrid"
            - "coco": Only keep COCO detections; drop custom-sourced
            - "custom": Only keep detections from CUSTOM_CLASSES (flower/fruit/tree/dumbbell)
            - "hybrid": Keep both; merge logic later
    
    Returns:
        Filtered detections
    """
    dets = canonicalize_final_detections(detections)
    
    if mode == "coco":
        # COCO-only: drop any custom-sourced detections
        out = []
        for d in dets:
            src = str(d.get('_source_model', '') or '').lower()
            if 'custom' in src:
                continue
            out.append(d)
        return out
    
    elif mode == "custom":
        # Custom4-only: keep only 4 custom classes
        out = []
        for d in dets:
            cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
            if cls_name not in CUSTOM_CLASSES:
                continue
            # Extra safety: never allow explicit COCO-sourced boxes in custom-only mode.
            src = str(d.get('_source_model', '') or '').lower()
            if ('custom' not in src) and any(k in src for k in ('coco', 'yolo')):
                continue
            out.append(d)
        return out
    
    elif mode == "hybrid":
        # Hybrid: keep all
        return dets
    
    else:
        return dets


def custom_has_specific_fruit_classes() -> bool:
    """Return True if the custom model defines any specific fruit classes."""
    try:
        custom_classes = get_custom_class_names()
        return any(f in custom_classes for f in SPECIFIC_FRUIT_CLASSES)
    except Exception:
        return False


def custom_has_any_fruit_class() -> bool:
    """Return True if the custom model defines fruit (generic or specific)."""
    try:
        custom_classes = get_custom_class_names()
        if 'fruit' in custom_classes:
            return True
        return any(f in custom_classes for f in SPECIFIC_FRUIT_CLASSES)
    except Exception:
        return False


def custom_prefers_generic_fruit() -> bool:
    """Custom model has generic 'fruit' but no specific fruit classes."""
    try:
        custom_classes = get_custom_class_names()
        return ('fruit' in custom_classes) and (not custom_has_specific_fruit_classes())
    except Exception:
        return False


def suppress_coco_fruit_when_custom_generic(detections: list) -> list:
    """Drop COCO specific fruit detections when custom uses generic 'fruit'."""
    dets = canonicalize_final_detections(detections)
    if not dets or not custom_prefers_generic_fruit():
        return dets
    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        if cls_name in SPECIFIC_FRUIT_CLASSES:
            src = str(d.get('_source_model', '') or '').lower()
            is_coco = ('coco' in src) or ('yolo' in src) or (src == 'coco_fruit_rescue')
            if is_coco:
                continue
        out.append(d)
    return out


def suppress_coco_fruit_when_custom_present(detections: list) -> list:
    """Drop COCO fruit detections when custom defines any fruit class."""
    dets = canonicalize_final_detections(detections)
    if not dets or not custom_has_any_fruit_class():
        return dets
    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        if cls_name == 'fruit' or cls_name in SPECIFIC_FRUIT_CLASSES:
            src = str(d.get('_source_model', '') or '').lower()
            is_coco = ('coco' in src) or ('yolo' in src) or (src == 'coco_fruit_rescue')
            if is_coco:
                continue
        out.append(d)
    return out


def build_hybrid_isolated_merge(coco_detections: list, custom_detections: list, img_w: int, img_h: int, base_conf: float) -> list:
    """Strict hybrid merge:
    - COCO contributes only COCO-domain classes (excluding custom domain).
    - Custom contributes only Custom4 classes.
    - Keep both sources together (e.g., person + flower).
    """
    custom_domain = set(get_custom_class_names()) | set(SPECIFIC_FRUIT_CLASSES) | {'potted plant'}

    coco_raw = ensure_source_model(canonicalize_final_detections(coco_detections), 'coco')
    coco_raw = filter_detections_by_mode(coco_raw, "coco")
    coco_keep = []
    for d in coco_raw:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        if cls_name in custom_domain:
            continue
        confv = float(d.get('conf', 0.0))
        if confv < max(0.20, float(base_conf) * 0.80):
            continue
        area_ratio, _, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)
        if area_ratio <= 0:
            continue
        if near_full and confv < 0.90:
            continue
        coco_keep.append(d)

    custom_raw = ensure_source_model(canonicalize_final_detections(custom_detections), 'custom')
    custom_keep = filter_detections_by_mode(custom_raw, "custom")

    merged = custom_keep + coco_keep
    merged = dedup_detections_by_class_nms_classwise(merged, default_iou=0.55)
    return merged


def suppress_cross_class_overlaps(detections: list, iou_thresh: float = 0.62, conf_gap: float = 0.10) -> list:
    """Drop lower-conf boxes when two different classes strongly overlap.

    Rationale: a single object sometimes receives multiple class labels
    (e.g., cong/curved shapes), which inflates counts. Keep the stronger
    hypothesis per location across classes.
    """
    dets = canonicalize_final_detections(detections)
    if not dets:
        return dets

    ordered = sorted(dets, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    kept = []
    for d in ordered:
        drop = False
        cls_d = normalize_class_name(d.get('class', '')) or 'unknown'
        for k in kept:
            cls_k = normalize_class_name(k.get('class', '')) or 'unknown'
            if cls_d == cls_k:
                continue
            # Only suppress overlaps within the target class set.
            if cls_d not in CROSS_CLASS_SUPPRESS_SET or cls_k not in CROSS_CLASS_SUPPRESS_SET:
                continue
            iou = _box_iou(d.get('xyxy'), k.get('xyxy'))
            if iou >= float(iou_thresh) and float(d.get('conf', 0.0)) <= float(k.get('conf', 0.0)) + float(conf_gap):
                drop = True
                break
        if not drop:
            kept.append(d)
    return kept


def resolve_cross_class_overlaps_with_priority(detections: list, iou_thresh: float = 0.60, conf_gap: float = 0.06) -> list:
    """Resolve cross-class overlaps using source and class priority bonuses.

    Keeps the highest scoring box where score = conf + source_priority + class_priority
    Only applies to pairs listed in CONFUSION_PAIRS to limit side-effects.
    """
    dets = canonicalize_final_detections(detections)
    if not dets:
        return dets

    def score_for(d):
        conf = float(d.get('conf', 0.0))
        src = str(d.get('_source_model', '') or '').lower().strip()
        src_key = 'custom' if 'custom' in src else ('coco' if 'coco' in src or 'yolo' in src or src == 'coco_fruit_rescue' else src)
        sbonus = float(SOURCE_PRIORITY.get(src_key, 0.0))
        cbonus = float(CLASS_PRIORITY.get(normalize_class_name(d.get('class', '')) or '', 0.0))
        return conf + sbonus + cbonus

    def source_conf_score_for(d):
        conf = float(d.get('conf', 0.0))
        src = str(d.get('_source_model', '') or '').lower().strip()
        src_key = 'custom' if 'custom' in src else ('coco' if 'coco' in src or 'yolo' in src or src == 'coco_fruit_rescue' else src)
        sbonus = float(SOURCE_PRIORITY.get(src_key, 0.0))
        return conf + sbonus


    def _is_specific_fruit(cls: str) -> bool:
        try:
            return (normalize_class_name(cls) or '') in SPECIFIC_FRUIT_CLASSES
        except Exception:
            return False

    def _is_fruit_like(cls: str) -> bool:
        cls_n = normalize_class_name(cls) or ''
        return (cls_n == 'fruit') or _is_specific_fruit(cls_n)

    orange_scene_n = int(
        sum(
            1
            for d in dets
            if (normalize_class_name(d.get('class', '')) or '') == 'orange'
        )
    )

    ordered = sorted(dets, key=lambda x: score_for(x), reverse=True)
    kept = []
    for d in ordered:
        drop = False
        cls_d = normalize_class_name(d.get('class', '')) or 'unknown'
        score_d = score_for(d)
        to_remove = []
        for k in kept:
            cls_k = normalize_class_name(k.get('class', '')) or 'unknown'
            if cls_d == cls_k:
                continue
            pair = frozenset((cls_d, cls_k))
            if pair not in CONFUSION_PAIRS:
                continue
            iou = _box_iou(d.get('xyxy'), k.get('xyxy'))
            # Special-case orange vs banana overlap to reduce label flicker.
            if pair == frozenset(('orange', 'banana')) and iou >= 0.40:
                # Preserve oranges in orange-heavy scenes.
                if orange_scene_n >= 3:
                    if cls_d == 'banana' and cls_k == 'orange':
                        drop = True
                        break
                    if cls_d == 'orange' and cls_k == 'banana':
                        to_remove.append(k)
                        continue
                score_src_d = source_conf_score_for(d)
                score_src_k = source_conf_score_for(k)
                if score_src_d > score_src_k:
                    to_remove.append(k)
                    continue
                drop = True
                break
            # special-case: flower vs fruit-like -> be more permissive and prefer fruit
            if ('flower' in pair) and (_is_fruit_like(cls_d) or _is_fruit_like(cls_k)):
                specific_iou = 0.40
                if iou >= float(specific_iou):
                    # identify which is flower and which is fruit-like
                    if cls_d == 'flower' and _is_fruit_like(cls_k):
                        flower_conf = float(d.get('conf', 0.0))
                        fruit_conf = float(k.get('conf', 0.0))
                        if fruit_conf >= flower_conf - 0.05:
                            drop = True
                            break
                    elif cls_k == 'flower' and _is_fruit_like(cls_d):
                        flower_conf = float(k.get('conf', 0.0))
                        fruit_conf = float(d.get('conf', 0.0))
                        if fruit_conf >= flower_conf - 0.05:
                            to_remove.append(k)
                            continue
            # special-case: dumbbell vs fruit-like -> stay conservative, but avoid
            # always sacrificing dumbbell when the model is only slightly less certain.
            if ('dumbbell' in pair) and (_is_fruit_like(cls_d) or _is_fruit_like(cls_k)):
                if iou >= 0.35:
                    # Prefer fruit only when it is meaningfully stronger; otherwise keep
                    # dumbbell candidates so downstream filters can validate them.
                    if cls_d == 'dumbbell' and _is_fruit_like(cls_k):
                        dumb_conf = float(d.get('conf', 0.0))
                        fruit_conf = float(k.get('conf', 0.0))
                        if fruit_conf >= dumb_conf + 0.05:
                            drop = True
                            break
                        if dumb_conf >= fruit_conf + 0.08:
                            to_remove.append(k)
                            continue
                    elif cls_k == 'dumbbell' and _is_fruit_like(cls_d):
                        dumb_conf = float(k.get('conf', 0.0))
                        fruit_conf = float(d.get('conf', 0.0))
                        if fruit_conf >= dumb_conf + 0.05:
                            to_remove.append(k)
                            continue
                        if dumb_conf >= fruit_conf + 0.08:
                            drop = True
                            break
            # default behavior: use score-based logic at provided iou_thresh
            if iou >= float(iou_thresh):
                score_k = score_for(k)
                if score_d <= score_k + float(conf_gap):
                    drop = True
                    break
        if not drop:
            if to_remove:
                for k in to_remove:
                    try:
                        kept.remove(k)
                    except ValueError:
                        pass
            kept.append(d)
    return kept


def suppress_cross_model_same_class_overlaps(
    detections: list,
    iou_thresh: float = 0.50,
    conf_gap: float = 0.05,
) -> list:
    """Drop duplicate boxes of the same class when they come from different models."""
    if not detections:
        return []
    dets = []
    for d in detections or []:
        if not isinstance(d, dict):
            continue
        d2 = dict(d)
        d2['class'] = normalize_class_name(d.get('class', '')) or 'unknown'
        dets.append(d2)
    if not dets:
        return []

    custom_classes = get_custom_class_names()
    ordered = sorted(dets, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    kept = []
    for d in ordered:
        drop = False
        cls_d = d.get('class', 'unknown')
        src_d = d.get('_source_model', None)
        for k in kept:
            if cls_d != k.get('class', 'unknown'):
                continue
            if cls_d not in custom_classes:
                continue
            src_k = k.get('_source_model', None)
            if not src_d or not src_k or src_d == src_k:
                continue
            iou = _box_iou(d.get('xyxy'), k.get('xyxy'))
            if iou >= float(iou_thresh) and float(d.get('conf', 0.0)) <= float(k.get('conf', 0.0)) + float(conf_gap):
                drop = True
                break
        if not drop:
            kept.append(d)
    return kept


def suppress_generic_fruit_overlaps(
    detections: list,
    iou_thresh: float = 0.40,
    min_specific_conf: float = 0.65,
    conf_gap: float = 0.08,
) -> list:
    """Prefer specific fruits (orange/banana/apple) over generic fruit on the same region.

    Rule:
    - If generic fruit overlaps specific fruit >= iou_thresh and specific conf >= min_specific_conf,
      drop generic fruit.
    - If overlapping specifics are all low-confidence, keep generic fruit only when it is
      significantly stronger (generic >= max_specific + conf_gap), otherwise drop generic fruit
      to avoid double counting.
    """
    dets = canonicalize_final_detections(detections)
    if not dets:
        return dets

    target_specific = {'orange', 'banana', 'apple'}
    specific = [
        d for d in dets
        if (normalize_class_name(d.get('class', '')) or '') in target_specific
    ]
    if not specific:
        return dets

    prefer_generic = custom_prefers_generic_fruit()

    def _is_coco_src(det: dict) -> bool:
        src = str(det.get('_source_model', '') or '').lower()
        return ('coco' in src) or ('yolo' in src) or (src == 'coco_fruit_rescue')

    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'

        # In custom-generic mode, drop COCO-specific fruits that overlap a generic fruit.
        if prefer_generic and cls_name in target_specific and _is_coco_src(d):
            overlap_generic = any(
                (normalize_class_name(g.get('class', '')) or '') == 'fruit'
                and _box_iou(d.get('xyxy'), g.get('xyxy')) >= float(iou_thresh)
                for g in dets
            )
            if overlap_generic:
                continue

        if cls_name != 'fruit':
            out.append(d)
            continue

        # In custom-generic mode, keep generic fruit unless there is strong custom-specific evidence.
        if prefer_generic:
            out.append(d)
            continue

        d_conf = float(d.get('conf', 0.0))
        overlap_specific = [
            s for s in specific
            if _box_iou(d.get('xyxy'), s.get('xyxy')) >= float(iou_thresh)
        ]
        if not overlap_specific:
            out.append(d)
            continue

        max_specific_conf = max(float(s.get('conf', 0.0)) for s in overlap_specific)
        if max_specific_conf >= float(min_specific_conf):
            # Strong specific fruit exists on the same object area -> drop generic fruit.
            continue

        # Specific fruit exists but is weak; keep generic fruit only if clearly stronger.
        if d_conf >= max_specific_conf + float(conf_gap):
            out.append(d)
            continue

        # Default for low-confidence overlaps: keep specific label only to avoid duplicates.
    return out


def _class_count_map(detections: list) -> dict:
    counts = {}
    for d in detections or []:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts


def _append_finalize_trace_log(stage_stats: dict, stage_detections: dict | None = None):
    """Append finalize debug stats to infer_log.json for offline trace checks."""
    if not isinstance(stage_stats, dict):
        return
    try:
        log_path = os.environ.get(
            'INFER_LOG_PATH',
            str(PROJECT_ROOT / 'tmp_test_outputs' / 'infer_log.json'),
        )
        payload = {
            'timestamp': datetime.now().isoformat(),
            'stage_stats': stage_stats,
            'stage_detections_counts': {
                k: (len(v) if isinstance(v, list) else 0)
                for k, v in (stage_detections or {}).items()
            },
        }

        if os.path.exists(log_path):
            try:
                raw = json.loads(Path(log_path).read_text(encoding='utf-8'))
            except Exception:
                raw = {}
        else:
            raw = {}

        if not isinstance(raw, dict):
            raw = {}
        items = raw.get('items')
        if not isinstance(items, list):
            items = []
        items.append(payload)
        if len(items) > 5000:
            items = items[-5000:]
        raw['items'] = items

        summary = raw.get('summary')
        if not isinstance(summary, dict):
            summary = {}
        summary['total_images'] = int(summary.get('total_images', 0)) + 1
        summary['total_detections'] = int(summary.get('total_detections', 0)) + int(stage_stats.get('final', 0) or 0)
        raw['summary'] = summary

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(raw, indent=2), encoding='utf-8')
    except Exception:
        pass


def suppress_conflicting_specific_fruits(
    detections: list,
    iou_thresh: float = 0.40,
    conf_gap: float = 0.05,
) -> list:
    """When multiple specific fruit labels overlap the same object, keep the strongest."""
    dets = canonicalize_final_detections(detections)
    if not dets:
        return dets

    fruits = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') in SPECIFIC_FRUIT_CLASSES]
    if len(fruits) < 2:
        return dets

    ordered = sorted(fruits, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    kept = []
    for d in ordered:
        drop = False
        for k in kept:
            if _box_iou(d.get('xyxy'), k.get('xyxy')) >= iou_thresh:
                if float(d.get('conf', 0.0)) <= float(k.get('conf', 0.0)) + conf_gap:
                    drop = True
                    break
        if not drop:
            kept.append(d)

    non_fruit = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') not in SPECIFIC_FRUIT_CLASSES]
    return non_fruit + kept


def finalize_frame_detections_for_count(
    detections: list,
    img_w: int,
    img_h: int,
    min_conf: float = MIN_COUNT_CONFIDENCE,
    debug: bool = False,
) -> list:
    """Finalize detections for counting in one frame.

    Rules:
    - confidence >= min_conf
    - valid box only
    - class-wise NMS/dedup
    - one object -> one box post-processing
    - no cross-class flower confusion
    """
    dets = canonicalize_final_detections(detections)
    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        confv = float(d.get('conf', 0.0))
        conf_floor = float(min_conf)
        if cls_name == 'tree':
            # Keep tree recall in custom mode while still filtering weak scene-level ghosts.
            src = str(d.get('_source_model', '') or '').lower()
            if 'custom' in src:
                conf_floor = max(0.18, float(min_conf) * 0.70)
            else:
                conf_floor = max(0.22, float(min_conf) * 0.80)
        elif cls_name == 'dumbbell':
            conf_floor = max(0.18, DUMBBELL_MIN_CONF, float(min_conf) * 0.55)
        # Custom flower: keep moderate confidence, rely on shape/visual pruning later.
        if cls_name == 'flower':
            conf_floor = max(0.24, float(min_conf) * 0.80, float(conf_floor))
        # Fruit should not be over-pruned; keep moderate conf floor.
        if cls_name == 'fruit' or cls_name in SPECIFIC_FRUIT_CLASSES:
            conf_floor = max(0.26, float(min_conf) * 0.75, float(conf_floor))
        if confv < conf_floor:
            continue
        area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if area_ratio <= 0:
            continue
        # Reject very large weak boxes that usually represent scene-level hallucinations.
        if cls_name == 'flower' and area_ratio >= 0.55 and confv < 0.82:
            continue
        if (cls_name == 'fruit' or cls_name in SPECIFIC_FRUIT_CLASSES) and area_ratio >= 0.45 and confv < 0.60:
            continue
        if cls_name == 'tree' and area_ratio >= 0.90 and confv < 0.72:
            continue
        if cls_name == 'dumbbell' and area_ratio >= 0.30 and confv < 0.55:
            continue
        if cls_name == 'tree' and area_ratio < 0.004 and confv < 0.40:
            continue
        out.append(d)

    # Per-class dedup tuning:
    # - lower IoU for sparse flower/tree to reduce duplicate boxes
    # - higher IoU for dense fruit clusters to keep nearby objects
    grouped = {}
    for d in out:
        cls = normalize_class_name(d.get('class', '')) or 'unknown'
        grouped.setdefault(cls, []).append(d)

    merged_out = []
    for cls, dets in grouped.items():
        if cls == 'flower':
            iou_thresh_cls = 0.52
        elif cls == 'tree':
            iou_thresh_cls = 0.50
        elif cls == 'dumbbell':
            iou_thresh_cls = 0.50
        elif cls == 'fruit' or cls in SPECIFIC_FRUIT_CLASSES:
            iou_thresh_cls = 0.68
        else:
            iou_thresh_cls = 0.60
        ordered = sorted(dets, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
        kept = []
        for d in ordered:
            overlap = False
            for k in kept:
                if _box_iou(d.get('xyxy'), k.get('xyxy')) >= float(iou_thresh_cls):
                    overlap = True
                    break
            if not overlap:
                kept.append(d)
        merged_out.extend(kept)
    out = merged_out

    # Tree duplicate cleanup: keep one representative per local canopy/trunk region.
    tree_boxes = [
        d for d in out
        if (normalize_class_name(d.get('class', '')) or '') == 'tree'
    ]
    if len(tree_boxes) > 1:
        tree_ordered = sorted(tree_boxes, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
        tree_kept = []
        for d in tree_ordered:
            g_d = _box_center_and_diag(d.get('xyxy'))
            if g_d is None:
                continue
            cx_d, cy_d, diag_d = g_d
            d_area, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
            if d_area <= 0:
                continue
            duplicated = False
            for k in tree_kept:
                iou = _box_iou(d.get('xyxy'), k.get('xyxy'))
                g_k = _box_center_and_diag(k.get('xyxy'))
                if g_k is None:
                    continue
                cx_k, cy_k, diag_k = g_k
                k_area, _, _ = _box_metrics(k.get('xyxy'), img_w, img_h)
                if k_area <= 0:
                    continue
                dist = math.hypot(cx_d - cx_k, cy_d - cy_k)
                area_ratio = d_area / max(1e-12, k_area)
                if iou >= 0.30:
                    duplicated = True
                    break
                if dist <= max(14.0, 0.20 * min(diag_d, diag_k)) and 0.35 <= area_ratio <= 2.80:
                    duplicated = True
                    break
            if not duplicated:
                tree_kept.append(d)

        if tree_kept:
            if any(float(t.get('conf', 0.0)) >= 0.70 for t in tree_kept):
                tree_kept = [t for t in tree_kept if float(t.get('conf', 0.0)) >= 0.30]
            non_tree = [
                d for d in out
                if (normalize_class_name(d.get('class', '')) or '') != 'tree'
            ]
            out = non_tree + tree_kept

    # In dumbbell-focused scenes, suppress weak incidental tree hallucinations.
    has_strong_dumbbell_pre = any(
        (normalize_class_name(d.get('class', '')) or '') == 'dumbbell'
        and float(d.get('conf', 0.0)) >= 0.28
        for d in out
    )
    if has_strong_dumbbell_pre:
        filtered_pre = []
        for d in out:
            cls_name = normalize_class_name(d.get('class', '')) or ''
            if cls_name != 'tree':
                filtered_pre.append(d)
                continue
            confv = float(d.get('conf', 0.0))
            area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
            if confv < 0.42 and area_ratio < 0.12:
                continue
            filtered_pre.append(d)
        out = filtered_pre

    # Snapshot before any cross-class / scene-level suppression so we can
    # fallback if later rules over-aggressively drop everything.
    pre_finalize = list(out)
    stage_stats = None
    stage_detections = None
    if debug:
        stage_stats = {
            'after_nms': len(out),
            'after_resolve_cross_class': None,
            'after_cross_class_suppress': None,
            'after_generic_fruit_overlap': None,
            'after_specific_fruit_conflict': None,
            'after_scene_level': None,
            'after_one_object_box': None,
            'after_dense_normalize': None,
            'after_suppress_flower_cross_class_confusions': None,
            'after_scene_rule_dedup': None,
            'after_scene_rules': None,
            'after_flower_pruning': None,
            'final': None,
            'fallback_used': False,
            'class_counts': {
                'after_nms': _class_count_map(out),
            },
        }
        # capture canonicalized detection snapshots per stage
        stage_detections = {
            'after_nms': [dict(d) for d in pre_finalize],
            'after_resolve_cross_class': None,
            'after_cross_class_suppress': None,
            'after_generic_fruit_overlap': None,
            'after_specific_fruit_conflict': None,
            'after_scene_level': None,
            'after_one_object_box': None,
            'after_dense_normalize': None,
            'after_suppress_flower_cross_class_confusions': None,
            'after_scene_rule_dedup': None,
            'after_flower_pruning': None,
            'after_scene_rules': None,
            'final': None,
        }

    # Resolve sensitive cross-class overlaps favoring source/class priority
    out = resolve_cross_class_overlaps_with_priority(out, iou_thresh=0.60, conf_gap=0.06)
    if debug and stage_stats is not None:
        stage_stats['after_resolve_cross_class'] = len(out)
        stage_stats['class_counts']['after_resolve_cross_class'] = _class_count_map(out)
        stage_detections['after_resolve_cross_class'] = [dict(d) for d in out]
    # Reduce cross-class double counting when the same object is labeled differently.
    out = suppress_cross_class_overlaps(out, iou_thresh=0.62, conf_gap=0.10)
    if debug and stage_stats is not None:
        stage_stats['after_cross_class_suppress'] = len(out)
        stage_stats['class_counts']['after_cross_class_suppress'] = _class_count_map(out)
        stage_detections['after_cross_class_suppress'] = [dict(d) for d in out]
    # Prefer specific fruit labels over generic fruit for the same object.
    out = suppress_generic_fruit_overlaps(out, iou_thresh=0.40, min_specific_conf=0.65, conf_gap=0.08)
    if debug and stage_stats is not None:
        stage_stats['after_generic_fruit_overlap'] = len(out)
        stage_stats['class_counts']['after_generic_fruit_overlap'] = _class_count_map(out)
        stage_detections['after_generic_fruit_overlap'] = [dict(d) for d in out]
    # Resolve conflicts between specific fruit labels (e.g., orange vs apple) on the same object.
    out = suppress_conflicting_specific_fruits(out, iou_thresh=0.40, conf_gap=0.05)
    if debug and stage_stats is not None:
        stage_stats['after_specific_fruit_conflict'] = len(out)
        stage_stats['class_counts']['after_specific_fruit_conflict'] = _class_count_map(out)
        stage_detections['after_specific_fruit_conflict'] = [dict(d) for d in out]

    if img_w > 0 and img_h > 0:
        out = suppress_scene_level_boxes(out, img_w=img_w, img_h=img_h)
        if debug and stage_stats is not None:
            stage_stats['after_scene_level'] = len(out)
            stage_stats['class_counts']['after_scene_level'] = _class_count_map(out)
            stage_detections['after_scene_level'] = [dict(d) for d in out]
        out = enforce_one_object_one_box(out, img_w=img_w, img_h=img_h)
        if debug and stage_stats is not None:
            stage_stats['after_one_object_box'] = len(out)
            stage_stats['class_counts']['after_one_object_box'] = _class_count_map(out)
            stage_detections['after_one_object_box'] = [dict(d) for d in out]
        out = normalize_dense_flower_boxes(out, img_w=img_w, img_h=img_h)
        if debug and stage_stats is not None:
            stage_stats['after_dense_normalize'] = len(out)
            stage_stats['class_counts']['after_dense_normalize'] = _class_count_map(out)
            stage_detections['after_dense_normalize'] = [dict(d) for d in out]
        out = collapse_dense_flower_duplicates(out, img_w=img_w, img_h=img_h)

        flower_n = int(sum(1 for d in out if (normalize_class_name(d.get('class', '')) or '') == 'flower'))
        flower_ratio = (float(flower_n) / float(max(1, len(out)))) if out else 0.0
        apply_flower_dense_rules = (flower_n >= 5 and flower_ratio > 0.65)

        prune_rejections = {} if debug else None
        if apply_flower_dense_rules:
            out = prune_dense_flower_noise(
                out,
                img_w=img_w,
                img_h=img_h,
                min_conf=float(min_conf),
                debug_rejections=prune_rejections,
            )
        elif debug and prune_rejections is not None:
            prune_rejections.update(
                {
                    'skipped': True,
                    'reason': 'gate_not_met',
                    'flower_count': flower_n,
                    'flower_ratio': float(flower_ratio),
                }
            )
        if debug and stage_stats is not None:
            stage_stats['after_flower_pruning'] = len(out)
            stage_stats['class_counts']['after_flower_pruning'] = _class_count_map(out)
            stage_stats['flower_prune_rejections'] = prune_rejections or {}
            stage_detections['after_flower_pruning'] = [dict(d) for d in out]
        if not apply_flower_dense_rules:
            # Sparse scene: remove obvious duplicates/tails without collapsing distinct flowers.
            out = aggressive_single_flower_dedup(out, img_w=img_w, img_h=img_h)
            out = prune_single_flower_tail_noise(out, img_w=img_w, img_h=img_h)
        tree_drop_reasons = {} if (debug and stage_stats is not None) else None
        tree_before_scene = int(
            sum(1 for d in out if (normalize_class_name(d.get('class', '')) or '') == 'tree')
        )
        has_strong_dumbbell = any(
            (normalize_class_name(d.get('class', '')) or '') == 'dumbbell'
            and float(d.get('conf', 0.0)) >= 0.30
            for d in out
        )
        fruit_refs_scene = [
            d
            for d in out
            if ((normalize_class_name(d.get('class', '')) or '') == 'fruit' or (normalize_class_name(d.get('class', '')) or '') in SPECIFIC_FRUIT_CLASSES)
            and float(d.get('conf', 0.0)) >= 0.45
        ]
        fruit_heavy_scene = len(fruit_refs_scene) >= 2

        scene_rescue = [
            dict(d)
            for d in out
            if (
                (normalize_class_name(d.get('class', '')) or '') in {'tree', 'dumbbell'}
                and not (
                    (normalize_class_name(d.get('class', '')) or '') == 'tree'
                    and fruit_heavy_scene
                    and not has_strong_dumbbell
                )
            )
            and float(d.get('conf', 0.0)) >= 0.22
        ]
        out = prune_dominant_flower_children(out, img_w=img_w, img_h=img_h)
        if tree_drop_reasons is not None:
            tree_after = int(sum(1 for d in out if (normalize_class_name(d.get('class', '')) or '') == 'tree'))
            if tree_before_scene > tree_after:
                tree_drop_reasons['prune_dominant_flower_children'] = int(tree_before_scene - tree_after)
            tree_before_scene = tree_after
        out = collapse_compact_flower_cluster(out, img_w=img_w, img_h=img_h)
        if tree_drop_reasons is not None:
            tree_after = int(sum(1 for d in out if (normalize_class_name(d.get('class', '')) or '') == 'tree'))
            if tree_before_scene > tree_after:
                tree_drop_reasons['collapse_compact_flower_cluster'] = int(tree_before_scene - tree_after)
            tree_before_scene = tree_after
        out = suppress_flower_cross_class_confusions(out, img_w=img_w, img_h=img_h, base_conf=float(min_conf))
        if debug and stage_stats is not None:
            stage_stats['after_suppress_flower_cross_class_confusions'] = len(out)
            stage_stats['class_counts']['after_suppress_flower_cross_class_confusions'] = _class_count_map(out)
            stage_detections['after_suppress_flower_cross_class_confusions'] = [dict(d) for d in out]
        if tree_drop_reasons is not None:
            tree_after = int(sum(1 for d in out if (normalize_class_name(d.get('class', '')) or '') == 'tree'))
            if tree_before_scene > tree_after:
                tree_drop_reasons['suppress_flower_cross_class_confusions'] = int(tree_before_scene - tree_after)
            tree_before_scene = tree_after
        # Keep dense overlapping flowers: class-wise NMS is safer than global same-class NMS.
        out = dedup_detections_by_class_nms_classwise(out, default_iou=0.70)
        tree_now = int(sum(1 for d in out if (normalize_class_name(d.get('class', '')) or '') == 'tree'))
        dumbbell_now = int(sum(1 for d in out if (normalize_class_name(d.get('class', '')) or '') == 'dumbbell'))
        if (tree_now == 0 or dumbbell_now == 0) and scene_rescue:
            add_back = [
                d for d in scene_rescue
                if ((normalize_class_name(d.get('class', '')) or '') == 'tree' and tree_now == 0)
                or ((normalize_class_name(d.get('class', '')) or '') == 'dumbbell' and dumbbell_now == 0)
            ]
            if add_back:
                out = dedup_detections_by_class_nms_classwise(out + add_back, default_iou=0.60)
                if debug and stage_stats is not None:
                    stage_stats['scene_rescue_readded'] = len(add_back)
        if debug and stage_stats is not None:
            stage_stats['after_scene_rule_dedup'] = len(out)
            stage_stats['class_counts']['after_scene_rule_dedup'] = _class_count_map(out)
            stage_detections['after_scene_rule_dedup'] = [dict(d) for d in out]

        # Late guard: in fruit-heavy scenes, remove weak/overlapping tree boxes.
        if img_w > 0 and img_h > 0:
            fruit_refs_late = [
                d
                for d in out
                if ((normalize_class_name(d.get('class', '')) or '') == 'fruit' or (normalize_class_name(d.get('class', '')) or '') in SPECIFIC_FRUIT_CLASSES)
                and float(d.get('conf', 0.0)) >= 0.40
            ]
            if len(fruit_refs_late) >= 2:
                before_late_guard = len(out)
                reduced = []
                dropped_tree_late = 0
                for d in out:
                    cls_name = normalize_class_name(d.get('class', '')) or ''
                    if cls_name != 'tree':
                        reduced.append(d)
                        continue
                    confv = float(d.get('conf', 0.0))
                    area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
                    overlap_fruit = any(_box_iou(d.get('xyxy'), f.get('xyxy')) >= 0.05 for f in fruit_refs_late)
                    if (confv < 0.78 and overlap_fruit) or (confv < 0.62 and area_ratio < 0.16):
                        dropped_tree_late += 1
                        continue
                    reduced.append(d)
                if dropped_tree_late > 0:
                    out = reduced
                    out = dedup_detections_by_class_nms_classwise(out, default_iou=0.60)
                if debug and stage_stats is not None:
                    stage_stats['after_late_fruit_tree_guard'] = len(out)
                    stage_stats['class_counts']['after_late_fruit_tree_guard'] = _class_count_map(out)
                    stage_detections['after_late_fruit_tree_guard'] = [dict(d) for d in out]
                    stage_stats['late_fruit_tree_guard_dropped'] = int(max(0, before_late_guard - len(out)))
        if tree_drop_reasons is not None:
            tree_after = int(sum(1 for d in out if (normalize_class_name(d.get('class', '')) or '') == 'tree'))
            if tree_before_scene > tree_after:
                tree_drop_reasons['scene_rules_classwise_dedup'] = int(tree_before_scene - tree_after)
            stage_stats['tree_drop_reasons_after_scene_rules'] = tree_drop_reasons
        if debug and stage_stats is not None:
            stage_stats['after_scene_rules'] = len(out)
            stage_stats['class_counts']['after_scene_rules'] = _class_count_map(out)
            stage_detections['after_scene_rules'] = [dict(d) for d in out]

    # Final stage stat
    if debug and stage_stats is not None:
        stage_stats['final'] = len(out)
        stage_stats['class_counts']['final'] = _class_count_map(out)
        stage_detections['final'] = [dict(d) for d in out]

    # Fallback: if all detections got dropped but pre_finalize had custom classes,
    # return the pre_finalize set to avoid total loss from aggressive rules.
    try:
        if (not out) and pre_finalize:
            custom_present = any(
                (normalize_class_name(d.get('class', '')) or '') in {'flower', 'fruit', 'tree', 'dumbbell'}
                for d in pre_finalize
            )
            if custom_present:
                out = pre_finalize
                if debug and stage_stats is not None:
                    stage_stats['fallback_used'] = True
                    stage_stats['final'] = len(out)
                    stage_stats['class_counts']['final'] = _class_count_map(out)
                    stage_detections['final'] = [dict(d) for d in out]
    except Exception:
        pass

    if debug and stage_stats is not None:
        _append_finalize_trace_log(stage_stats, stage_detections)
        return out, stage_stats, stage_detections

    return out


def aggressive_single_flower_dedup(detections: list, img_w: int, img_h: int) -> list:
    """For single/double flower cases, keep only the highest confidence box.
    
    Ảnh 1 hoa nhưng 2+ boxes → giữ box confidence cao nhất.
    """
    dets = canonicalize_final_detections(detections)
    if not dets:
        return dets
    
    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    
    # Only apply aggressive dedup when we have 2-3 flowers with very high overlap
    if len(flowers) < 2 or len(flowers) > 3:
        return dets
    
    # Check if flowers are highly overlapped (likely duplicates of same object)
    ordered = sorted(flowers, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    top_conf = float(ordered[0].get('conf', 0.0))
    second_conf = float(ordered[1].get('conf', 0.0)) if len(ordered) > 1 else 0.0
    
    # Only aggressive dedup if top box is clearly dominant (>= 0.80 conf)
    # and other boxes are much weaker (< 0.70 conf)
    if top_conf >= 0.86 and second_conf < 0.64:
        # Check overlap with top box
        top_iou_with_second = _box_iou(ordered[0].get('xyxy'), ordered[1].get('xyxy'))
        if top_iou_with_second >= 0.72:
            # Remove weak duplicates, keep only strongest
            non_flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
            return non_flowers + [ordered[0]]
    
    return dets


def prune_single_flower_tail_noise(detections: list, img_w: int, img_h: int) -> list:
    """Remove weak tail boxes in single-flower scenes (e.g., stem-only ghost boxes).

    Typical failure case: one dominant flower box + one weak small box at the stem.
    """
    dets = canonicalize_final_detections(detections)
    if not dets or img_w <= 0 or img_h <= 0:
        return dets

    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    if len(flowers) < 2 or len(flowers) > 3:
        return dets

    ordered = sorted(flowers, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    best = ordered[0]
    best_conf = float(best.get('conf', 0.0))
    if best_conf < 0.90:
        return dets

    best_area, _, _ = _box_metrics(best.get('xyxy'), img_w, img_h)
    best_geom = _box_center_and_diag(best.get('xyxy'))
    if best_area <= 0 or best_geom is None:
        return dets
    bcx, bcy, bdiag = best_geom

    kept = [best]
    for d in ordered[1:]:
        confv = float(d.get('conf', 0.0))
        area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        iou_best = _box_iou(d.get('xyxy'), best.get('xyxy'))
        geom = _box_center_and_diag(d.get('xyxy'))
        if geom is None:
            continue
        cx, cy, _ = geom
        dist = math.hypot(cx - bcx, cy - bcy)

        # Weak tail boxes: remove aggressively.
        if confv < max(0.50, best_conf - 0.25):
            continue
        if area_ratio < (0.35 * best_area) and iou_best < 0.20:
            continue
        if dist > (0.45 * bdiag) and confv < 0.80:
            continue

        kept.append(d)

    non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    return non_flower + kept


def collapse_sparse_flower_duplicates(detections: list, img_w: int, img_h: int) -> list:
    """Final strict duplicate collapse for flower boxes in sparse scenes."""
    dets = canonicalize_final_detections(detections)
    if not dets or img_w <= 0 or img_h <= 0:
        return dets

    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    if len(flowers) < 2:
        return dets
    flower_ratio = float(len(flowers)) / float(max(1, len(dets)))
    if flower_ratio <= 0.55:
        return dets

    # This pass is for sparse close-up duplicates only.
    # Do not run it for medium/dense flower groups.
    if len(flowers) > 6:
        return dets

    # Keep dense tiny-flower scenes untouched to avoid under-counting real neighbors.
    if len(flowers) > 18:
        return dets

    flower_areas = []
    for d in flowers:
        area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if area_ratio > 0:
            flower_areas.append(area_ratio)
    if not flower_areas:
        return dets

    med_area = sorted(flower_areas)[len(flower_areas) // 2]
    if med_area < 0.003 and len(flowers) > 8:
        return dets

    ordered = sorted(flowers, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    kept = []
    for d in ordered:
        geom_d = _box_center_and_diag(d.get('xyxy'))
        if geom_d is None:
            continue
        cx_d, cy_d, diag_d = geom_d
        d_area, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if d_area <= 0:
            continue

        try:
            dx1, dy1, dx2, dy2 = map(float, d.get('xyxy')[:4])
        except Exception:
            continue

        duplicated = False
        for k in kept:
            geom_k = _box_center_and_diag(k.get('xyxy'))
            if geom_k is None:
                continue
            cx_k, cy_k, diag_k = geom_k
            k_area, _, _ = _box_metrics(k.get('xyxy'), img_w, img_h)
            if k_area <= 0:
                continue

            try:
                kx1, ky1, kx2, ky2 = map(float, k.get('xyxy')[:4])
            except Exception:
                continue

            iou = _box_iou(d.get('xyxy'), k.get('xyxy'))

            inter_x1 = max(dx1, kx1)
            inter_y1 = max(dy1, ky1)
            inter_x2 = min(dx2, kx2)
            inter_y2 = min(dy2, ky2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            iom = inter / min(max(1e-12, d_area), max(1e-12, k_area))

            dist = math.hypot(cx_d - cx_k, cy_d - cy_k)
            min_diag = min(diag_d, diag_k)
            area_ratio = d_area / max(1e-12, k_area)

            # Containment: one center lies inside the other box.
            d_center_in_k = (kx1 <= cx_d <= kx2) and (ky1 <= cy_d <= ky2)
            k_center_in_d = (dx1 <= cx_k <= dx2) and (dy1 <= cy_k <= dy2)
            contained = d_center_in_k or k_center_in_d

            if iou >= 0.70:
                duplicated = True
                break
            if iom >= 0.88:
                duplicated = True
                break
            if dist <= max(8.0, 0.10 * min_diag) and 0.45 <= area_ratio <= 2.40 and (iou >= 0.45 or iom >= 0.75):
                duplicated = True
                break
            if contained and iou >= 0.40 and dist <= max(10.0, 0.18 * min_diag) and 0.40 <= area_ratio <= 2.70:
                duplicated = True
                break

        if not duplicated:
            kept.append(d)

    if not kept:
        return dets

    non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    return non_flower + kept


def collapse_sparse_large_flower_duplicates(detections: list, img_w: int, img_h: int) -> list:
    """Aggressive collapse for sparse scenes with large single flowers."""
    dets = canonicalize_final_detections(detections)
    if not dets or img_w <= 0 or img_h <= 0:
        return dets

    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    if len(flowers) < 2:
        return dets
    flower_ratio = float(len(flowers)) / float(max(1, len(dets)))
    if flower_ratio <= 0.55:
        return dets

    areas = []
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    for d in flowers:
        a, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if a > 0:
            areas.append(a)
        try:
            x1, y1, x2, y2 = map(float, d.get('xyxy')[:4])
        except Exception:
            continue
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
    if not areas:
        return dets

    med_area = sorted(areas)[len(areas) // 2]
    max_area = max(areas)
    union_area = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
    union_ratio = union_area / float(max(1.0, img_w * img_h))
    # Only apply when flowers are reasonably large (likely 1-2 big flowers).
    if med_area < 0.004:
        return dets
    # Skip wide-spread dense fields to avoid collapsing true multi-flower scenes.
    if len(flowers) >= 6 and union_ratio >= 0.50 and max_area <= 0.18:
        return dets
    # For larger counts, only collapse when there's a very dominant close-up flower.
    if len(flowers) > 6:
        if not (max_area >= 0.20 and union_ratio <= 0.50):
            return dets

    # For small sparse groups (2-4), only collapse when one flower is clearly dominant.
    if len(flowers) <= 4:
        ordered_areas = sorted(areas, reverse=True)
        top = ordered_areas[0]
        second = ordered_areas[1] if len(ordered_areas) > 1 else 0.0
        if not (top >= 0.045 and top >= max(0.010, second * 2.2)):
            return dets

    ordered = sorted(flowers, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    kept = []
    for d in ordered:
        geom_d = _box_center_and_diag(d.get('xyxy'))
        if geom_d is None:
            continue
        cx_d, cy_d, diag_d = geom_d
        d_area, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if d_area <= 0:
            continue
        try:
            dx1, dy1, dx2, dy2 = map(float, d.get('xyxy')[:4])
        except Exception:
            continue
        duplicated = False
        for k in kept:
            geom_k = _box_center_and_diag(k.get('xyxy'))
            if geom_k is None:
                continue
            cx_k, cy_k, diag_k = geom_k
            k_area, _, _ = _box_metrics(k.get('xyxy'), img_w, img_h)
            if k_area <= 0:
                continue
            iou = _box_iou(d.get('xyxy'), k.get('xyxy'))
            dist = math.hypot(cx_d - cx_k, cy_d - cy_k)
            area_ratio = d_area / max(1e-12, k_area)
            # Merge if strong overlap or centers are close for similarly sized boxes.
            if iou >= 0.25:
                duplicated = True
            if dist <= max(12.0, 0.22 * min(diag_d, diag_k)) and 0.45 <= area_ratio <= 2.50:
                duplicated = True
            if duplicated:
                break
        if not duplicated:
            kept.append(d)

    # Limit keep count for very large flowers to avoid multi-box duplicates.
    if max_area >= 0.06 or med_area >= 0.025:
        max_keep = 2
    elif med_area >= 0.012:
        max_keep = 2
    else:
        max_keep = 4
    kept = kept[:max_keep]

    non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    return non_flower + kept


def prune_dominant_flower_children(detections: list, img_w: int, img_h: int) -> list:
    """If one large flower dominates, drop small overlapping child boxes."""
    dets = canonicalize_final_detections(detections)
    if not dets or img_w <= 0 or img_h <= 0:
        return dets

    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    if len(flowers) < 3:
        return dets

    areas = []
    for d in flowers:
        a, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        areas.append(max(0.0, a))
    if not areas:
        return dets

    # Identify dominant large flower.
    ordered = sorted(zip(flowers, areas), key=lambda x: x[1], reverse=True)
    dominant, dom_area = ordered[0]
    second_area = ordered[1][1] if len(ordered) > 1 else 0.0
    if dom_area < 0.010:
        return dets
    if second_area >= 0.65 * dom_area:
        return dets  # likely multiple large flowers

    dom_geom = _box_center_and_diag(dominant.get('xyxy'))
    if dom_geom is None:
        return dets
    dcx, dcy, ddiag = dom_geom
    dom_conf = float(dominant.get('conf', 0.0))

    kept = []
    for d, a in ordered:
        geom_d = _box_center_and_diag(d.get('xyxy'))
        if geom_d is None:
            continue
        cx, cy, diag = geom_d
        dist = math.hypot(cx - dcx, cy - dcy)
        # Keep dominant and any far/large peer, drop small overlapping children.
        if d is dominant:
            kept.append(d)
            continue
        if a >= 0.35 * dom_area:
            kept.append(d)
            continue
        if dist >= max(18.0, 0.55 * ddiag):
            kept.append(d)
            continue
        if float(d.get('conf', 0.0)) >= dom_conf + 0.08:
            kept.append(d)
            continue

    non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    return non_flower + kept


def collapse_compact_flower_cluster(detections: list, img_w: int, img_h: int) -> list:
    """Collapse compact clusters of many flower boxes into 1-2 boxes."""
    dets = canonicalize_final_detections(detections)
    if not dets or img_w <= 0 or img_h <= 0:
        return dets

    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    if len(flowers) < 3:
        return dets

    centers = []
    areas = []
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    for d in flowers:
        geom = _box_center_and_diag(d.get('xyxy'))
        if geom is None:
            continue
        cx, cy, _ = geom
        centers.append((cx, cy))
        a, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if a > 0:
            areas.append(a)
        try:
            x1, y1, x2, y2 = map(float, d.get('xyxy')[:4])
        except Exception:
            continue
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
    if len(centers) < 3 or not areas:
        return dets

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs) / max(1, len(xs))
    var_y = sum((y - mean_y) ** 2 for y in ys) / max(1, len(ys))
    std_x = math.sqrt(var_x)
    std_y = math.sqrt(var_y)

    med_area = float(sorted(areas)[len(areas) // 2])
    max_area = float(max(areas))
    union_area = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
    union_ratio = union_area / float(max(1.0, img_w * img_h))

    # Compact cluster: many boxes tightly packed in one region.
    compact = (std_x <= 0.20 * img_w and std_y <= 0.20 * img_h)
    if compact and len(flowers) >= 6:
        # Skip for dense fields of tiny flowers.
        if max_area < 0.010 and med_area < 0.003:
            return dets
        if union_ratio >= 0.50 and med_area < 0.004:
            return dets

        ordered = sorted(flowers, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
        clusters = []
        base_radius = 0.18 * min(img_w, img_h)
        if max_area >= 0.020:
            base_radius = 0.22 * min(img_w, img_h)
        for d in ordered:
            geom = _box_center_and_diag(d.get('xyxy'))
            if geom is None:
                continue
            cx, cy, _ = geom
            merged = False
            for c in clusters:
                rcx, rcy = c['center']
                dist = math.hypot(cx - rcx, cy - rcy)
                if dist <= base_radius:
                    c['items'].append(d)
                    if float(d.get('conf', 0.0)) > float(c['rep'].get('conf', 0.0)):
                        c['rep'] = d
                        c['center'] = (cx, cy)
                    merged = True
                    break
            if not merged:
                clusters.append({'rep': d, 'items': [d], 'center': (cx, cy)})

        if len(clusters) <= 2:
            kept_flowers = [dict(c['rep']) for c in clusters]
            if len(clusters) == 1 and (max_area >= 0.025 or med_area >= 0.008):
                kept_flowers = kept_flowers[:1]
            non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
            return non_flower + kept_flowers

    return dets


def collapse_dense_flower_duplicates(detections: list, img_w: int, img_h: int) -> list:
    """Collapse obvious duplicate flower boxes in dense scenes.

    This pass targets overcount (many near-identical boxes per flower) while
    preserving nearby real flowers.
    """
    dets = canonicalize_final_detections(detections)
    if not dets or img_w <= 0 or img_h <= 0:
        return dets

    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    if len(flowers) < 9:
        return dets

    ordered = sorted(flowers, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    kept = []
    for d in ordered:
        g_d = _box_center_and_diag(d.get('xyxy'))
        if g_d is None:
            continue
        cx_d, cy_d, diag_d = g_d
        try:
            dx1, dy1, dx2, dy2 = map(float, d.get('xyxy')[:4])
        except Exception:
            continue
        d_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
        if d_area <= 0:
            continue

        duplicate = False
        for k in kept:
            g_k = _box_center_and_diag(k.get('xyxy'))
            if g_k is None:
                continue
            cx_k, cy_k, diag_k = g_k
            try:
                kx1, ky1, kx2, ky2 = map(float, k.get('xyxy')[:4])
            except Exception:
                continue
            k_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
            if k_area <= 0:
                continue

            iou = _box_iou(d.get('xyxy'), k.get('xyxy'))
            inter_x1 = max(dx1, kx1)
            inter_y1 = max(dy1, ky1)
            inter_x2 = min(dx2, kx2)
            inter_y2 = min(dy2, ky2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            iom = inter / min(max(1e-12, d_area), max(1e-12, k_area))

            dist = math.hypot(cx_d - cx_k, cy_d - cy_k)
            min_diag = min(diag_d, diag_k)
            area_ratio = d_area / max(1e-12, k_area)

            if iou >= 0.30:
                duplicate = True
                break
            if iom >= 0.66:
                duplicate = True
                break
            if dist <= max(10.0, 0.20 * min_diag) and (iou >= 0.12 or iom >= 0.35) and 0.45 <= area_ratio <= 2.20:
                duplicate = True
                break

        if not duplicate:
            kept.append(d)

    if not kept:
        return dets

    # Second pass for dense overlap: if a flower center lies inside a stronger
    # flower box with similar size, treat it as the same object.
    kept2 = []
    ordered2 = sorted(kept, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    for d in ordered2:
        g_d = _box_center_and_diag(d.get('xyxy'))
        if g_d is None:
            continue
        cx_d, cy_d, _ = g_d
        try:
            dx1, dy1, dx2, dy2 = map(float, d.get('xyxy')[:4])
        except Exception:
            continue
        d_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
        if d_area <= 0:
            continue

        duplicate2 = False
        for k in kept2:
            try:
                kx1, ky1, kx2, ky2 = map(float, k.get('xyxy')[:4])
            except Exception:
                continue
            k_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
            if k_area <= 0:
                continue
            iou = _box_iou(d.get('xyxy'), k.get('xyxy'))
            inter_x1 = max(dx1, kx1)
            inter_y1 = max(dy1, ky1)
            inter_x2 = min(dx2, kx2)
            inter_y2 = min(dy2, ky2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            iom = inter / min(max(1e-12, d_area), max(1e-12, k_area))
            area_ratio = d_area / max(1e-12, k_area)
            center_inside = (kx1 <= cx_d <= kx2) and (ky1 <= cy_d <= ky2)
            if center_inside and (iou >= 0.06 or iom >= 0.45) and 0.40 <= area_ratio <= 2.50:
                duplicate2 = True
                break

        if not duplicate2:
            kept2.append(d)

    if kept2:
        kept = kept2

    # Guardrail: do not over-collapse dense scenes.
    if len(kept) < max(5, int(round(len(flowers) * 0.42))):
        return dets

    non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    return non_flower + kept


def prune_dense_flower_noise(
    detections: list,
    img_w: int,
    img_h: int,
    min_conf: float = MIN_COUNT_CONFIDENCE,
    debug_rejections: dict | None = None,
) -> list:
    """Prune noisy flower boxes in dense scenes using confidence + geometry."""
    dets = canonicalize_final_detections(detections)
    if not dets or img_w <= 0 or img_h <= 0:
        return dets

    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    if len(flowers) < 5:
        return dets

    flower_ratio = float(len(flowers)) / float(max(1, len(dets)))
    if flower_ratio <= 0.65:
        return dets

    # If flowers spread across a wide area (field/scene-wide), avoid aggressive pruning.
    try:
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        centers = []
        for d in flowers:
            x1, y1, x2, y2 = map(float, d.get('xyxy')[:4])
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            centers.append((cx, cy))
        union_area = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
        union_ratio = union_area / float(max(1.0, img_w * img_h))
        if centers:
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / len(xs))
            std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / len(ys))
        else:
            std_x = 0.0
            std_y = 0.0
        if (union_ratio >= 0.55 and len(flowers) >= 8) or (std_x >= 0.28 * img_w or std_y >= 0.28 * img_h):
            return dets
    except Exception:
        pass

    areas = []
    confs = []
    for d in flowers:
        a, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if a > 0:
            areas.append(a)
            confs.append(float(d.get('conf', 0.0)))
    if not areas:
        return dets

    med_area = float(sorted(areas)[len(areas) // 2])
    med_conf = float(sorted(confs)[len(confs) // 2]) if confs else float(min_conf)
    dense_conf_floor = max(0.44, float(min_conf) + 0.10, med_conf - 0.18)

    kept = []
    reject_counts = {
        'low_conf': 0,
        'small_area': 0,
        'large_area': 0,
        'bad_aspect': 0,
        'near_full': 0,
    }
    for d in flowers:
        confv = float(d.get('conf', 0.0))
        area_ratio, aspect, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)
        if area_ratio <= 0:
            continue
        if area_ratio < 0.0003 and confv >= 0.45:
            kept.append(d)
            continue
        if near_full:
            if debug_rejections is not None:
                reject_counts['near_full'] += 1
            continue
        if confv < dense_conf_floor:
            if debug_rejections is not None:
                reject_counts['low_conf'] += 1
            continue
        if area_ratio < max(0.00015, 0.30 * med_area):
            if debug_rejections is not None:
                reject_counts['small_area'] += 1
            continue
        if area_ratio > min(0.22, 3.20 * med_area) and confv < 0.95:
            if debug_rejections is not None:
                reject_counts['large_area'] += 1
            continue
        if aspect > 2.10 and confv < 0.94:
            if debug_rejections is not None:
                reject_counts['bad_aspect'] += 1
            continue
        kept.append(d)

    if debug_rejections is not None:
        debug_rejections.clear()
        debug_rejections.update(
            {
                'flower_candidates': len(flowers),
                'flower_kept': len(kept),
                'dense_conf_floor': float(dense_conf_floor),
                'median_area': float(med_area),
                'median_conf': float(med_conf),
                'rejected': reject_counts,
            }
        )

    if len(kept) < max(5, int(round(len(flowers) * 0.55))):
        return dets

    non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    return non_flower + kept


def prune_flower_boxes_by_visual_evidence(image_np: np.ndarray, detections: list) -> list:
    """Drop dense-scene flower false positives using ROI visual evidence.

    Target: remove stem/leaf boxes that the detector marks as flower.
    The check is conservative and only active for dense flower scenes.
    """
    dets = canonicalize_final_detections(detections)
    if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
        return dets
    h, w = image_np.shape[:2]
    if h <= 0 or w <= 0:
        return dets

    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    if not flowers:
        return dets
    # Only prune by visual evidence in clearly dense scenes.
    if len(flowers) < 10:
        return dets

    sparse_mode = len(flowers) <= 6

    def flower_visual_features(xyxy):
        try:
            x1, y1, x2, y2 = map(int, xyxy[:4])
        except Exception:
            return None

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 + 4 or y2 <= y1 + 6:
            return None

        roi = image_np[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        hch = hsv[:, :, 0]
        sch = hsv[:, :, 1]
        vch = hsv[:, :, 2]
        ycc = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
        ych = ycc[:, :, 0]
        cr = ycc[:, :, 1]
        cb = ycc[:, :, 2]
        skin_mask = (
            (ych >= 40)
            & (cr >= 135)
            & (cr <= 180)
            & (cb >= 85)
            & (cb <= 135)
            & (hch <= 25)
            & (sch >= 25)
        )

        non_green = ((hch < 35) | (hch > 95))
        colorful = (sch >= 55) & (vch >= 40)
        bright_white = (vch >= 170) & (sch <= 55)
        petal_like = ((non_green & colorful) | bright_white) & (~skin_mask)

        green_like = ((hch >= 35) & (hch <= 95) & (sch >= 40) & (vch >= 35))

        petal_ratio = float(np.count_nonzero(petal_like)) / float(max(1, roi.shape[0] * roi.shape[1]))
        green_ratio = float(np.count_nonzero(green_like)) / float(max(1, roi.shape[0] * roi.shape[1]))

        cy1 = int(max(0, roi.shape[0] * 0.20))
        cy2 = int(max(cy1 + 1, roi.shape[0] * 0.80))
        cx1 = int(max(0, roi.shape[1] * 0.20))
        cx2 = int(max(cx1 + 1, roi.shape[1] * 0.80))
        center_roi = petal_like[cy1:cy2, cx1:cx2]
        center_petal_ratio = float(np.count_nonzero(center_roi)) / float(max(1, center_roi.size))

        ys, xs = np.where(petal_like)
        if len(xs) == 0:
            return {
                'petal_ratio': petal_ratio,
                'green_ratio': green_ratio,
                'center_petal_ratio': center_petal_ratio,
                'petal_bbox': None,
                'petal_centroid_y': 1.0,
                'petal_centroid_x': 0.5,
                'petal_fill_ratio': 0.0,
                'petal_spread_x': 0.0,
                'petal_spread_y': 0.0,
            }

        px1 = int(xs.min())
        py1 = int(ys.min())
        px2 = int(xs.max()) + 1
        py2 = int(ys.max()) + 1
        pbox_area = max(1, (px2 - px1) * (py2 - py1))
        petal_fill_ratio = float(len(xs)) / float(pbox_area)
        petal_spread_x = float(px2 - px1) / float(max(1, roi.shape[1]))
        petal_spread_y = float(py2 - py1) / float(max(1, roi.shape[0]))
        petal_centroid_x = float(xs.mean()) / float(max(1, roi.shape[1]))
        petal_centroid_y = float(ys.mean()) / float(max(1, roi.shape[0]))
        return {
            'petal_ratio': petal_ratio,
            'green_ratio': green_ratio,
            'center_petal_ratio': center_petal_ratio,
            'petal_bbox': [px1, py1, px2, py2],
            'petal_centroid_y': petal_centroid_y,
            'petal_centroid_x': petal_centroid_x,
            'petal_fill_ratio': petal_fill_ratio,
            'petal_spread_x': petal_spread_x,
            'petal_spread_y': petal_spread_y,
        }

    kept = []
    for d in flowers:
        confv = float(d.get('conf', 0.0))
        xy = d.get('xyxy') or []
        if len(xy) < 4:
            continue
        try:
            x1, y1, x2, y2 = map(int, xy[:4])
        except Exception:
            continue

        area_ratio, aspect, _ = _box_metrics([x1, y1, x2, y2], w, h)
        if area_ratio <= 0:
            continue

        feats = flower_visual_features([x1, y1, x2, y2])
        if feats is None:
            continue
        petal_ratio = float(feats['petal_ratio'])
        center_petal_ratio = float(feats['center_petal_ratio'])
        green_ratio = float(feats['green_ratio'])
        petal_centroid_y = float(feats['petal_centroid_y'])
        petal_fill_ratio = float(feats['petal_fill_ratio'])
        petal_spread_x = float(feats['petal_spread_x'])
        petal_spread_y = float(feats['petal_spread_y'])

        # Sparse scenes: keep a single large/clear flower even if petal metrics are imperfect.
        if sparse_mode and area_ratio >= 0.01 and confv >= 0.70:
            kept.append(d)
            continue
        if sparse_mode and confv >= 0.62 and center_petal_ratio >= 0.06 and petal_ratio >= 0.05:
            kept.append(d)
            continue

        # RULE (from leaf_fp_candidates analysis): drop boxes that are likely
        # leaf/stem false positives: high green content, low petal evidence,
        # and small area.
        # Drop if: green_ratio > 0.45 AND petal_ratio < 0.15 AND area_ratio < 0.02
        try:
            if (green_ratio > 0.45) and (petal_ratio < 0.15) and (area_ratio < 0.02):
                continue
        except Exception:
            pass

        # Very strong predictions are kept unless geometry is clearly implausible.
        if confv >= 0.97 and aspect <= 2.8 and center_petal_ratio >= 0.08:
            kept.append(d)
            continue

        # Stem/leaf false boxes: tall, green-heavy, petals only near one edge.
        # PATCH: reduce flower FP / fruit confusion — tighten green ratio threshold
        if aspect > 1.7 and center_petal_ratio < 0.12 and petal_ratio < 0.18 and green_ratio > 0.42 and confv < 0.97:
            continue
        if petal_centroid_y < 0.28 and aspect > 1.35 and petal_spread_x < 0.42 and confv < 0.97:
            continue
        if petal_fill_ratio < 0.18 and petal_ratio < 0.12 and confv < 0.95:
            continue
        # Stronger sparse-mode pruning
        if sparse_mode and center_petal_ratio < 0.10 and petal_ratio < 0.15 and confv < 0.96:
            continue
        # PATCH: stricter center_petal_ratio requirement for moderate confidence
        if confv < 0.86 and center_petal_ratio < 0.12:
            continue
        if sparse_mode and center_petal_ratio < 0.12 and petal_ratio < 0.16 and confv < 0.92:
            continue

        if confv >= 0.86 and center_petal_ratio >= 0.10 and petal_ratio >= 0.07:
            kept.append(d)
            continue
        if confv >= 0.72 and center_petal_ratio >= 0.16 and petal_ratio >= 0.12:
            kept.append(d)
            continue

    # Guardrail: avoid over-pruning dense scenes; in sparse scenes allow stronger pruning.
    if (not sparse_mode) and len(kept) < max(4, int(round(len(flowers) * 0.50))):
        return dets

    non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    return non_flower + kept


def suppress_flower_on_fruit_confusions(detections: list, img_w: int, img_h: int, image_np: np.ndarray = None) -> list:
    """Drop flower boxes that likely overlap/duplicate fruit boxes.
    PATCH: reduce flower FP / fruit confusion — lower IoU threshold, tighten conf delta,
    and optionally use hue/saturation evidence from the fruit ROI when image is available.
    """
    dets = canonicalize_final_detections(detections)
    if not dets:
        return []

    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    fruits = [
        d for d in dets
        if (normalize_class_name(d.get('class', '')) or '') in (set(['fruit']) | set(SPECIFIC_FRUIT_CLASSES))
    ]
    if not flowers or not fruits:
        return dets

    fruit_refs = []
    for f in fruits:
        area_ratio, aspect, _ = _box_metrics(f.get('xyxy'), img_w, img_h)
        if area_ratio <= 0:
            continue
        fruit_refs.append((f, area_ratio, aspect, float(f.get('conf', 0.0))))

    fruit_scene = len(fruit_refs) >= 2
    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        if cls_name != 'flower':
            out.append(d)
            continue

        d_conf = float(d.get('conf', 0.0))
        d_area, d_aspect, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        drop = False
        for f, f_area, f_aspect, f_conf in fruit_refs:
            iou = _box_iou(d.get('xyxy'), f.get('xyxy'))
            area_ratio = d_area / max(1e-12, f_area)
            # Fruit-heavy hard guard: overlapping "flower" on fruit is usually confusion.
            if fruit_scene and iou >= 0.08 and 0.28 <= area_ratio <= 2.60:
                drop = True
                break
            # Prefer fruit labels (including specific fruits like banana) when overlapping.
            if iou >= 0.12 and 0.40 <= area_ratio <= 1.80 and d_conf <= f_conf + 0.08:
                drop = True
                break
            if iou >= 0.12 and f_conf >= 0.78 and f_area >= 0.010 and d_conf <= f_conf + 0.04:
                drop = True
                break

            # If image provided, prefer fruit when its hue/saturation indicate fruit color
            if not drop and image_np is not None:
                try:
                    # crop fruit ROI and compute mean hue/saturation
                    fx1, fy1, fx2, fy2 = map(int, f.get('xyxy')[:4])
                    fx1 = max(0, min(img_w - 1, fx1))
                    fx2 = max(0, min(img_w, fx2))
                    fy1 = max(0, min(img_h - 1, fy1))
                    fy2 = max(0, min(img_h, fy2))
                    if fx2 > fx1 and fy2 > fy1:
                        roi = image_np[fy1:fy2, fx1:fx2]
                        if roi.size > 0:
                            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                            hch = hsv[:, :, 0]
                            sch = hsv[:, :, 1]
                            # mean hue in OpenCV is 0..179
                            mean_h = float(np.mean(hch))
                            mean_s = float(np.mean(sch))
                            # fruit hue range approx 5..30 (orange/yellow)
                            if 5.0 <= mean_h <= 30.0 and mean_s >= 90.0:
                                # favor fruit label strongly
                                if iou >= 0.08 and d_conf <= f_conf + 0.12:
                                    drop = True
                                    break
                except Exception:
                    pass

        if not drop:
            out.append(d)
    return out


def suppress_face_like_flower_boxes(image_np: np.ndarray, detections: list, img_w: int, img_h: int) -> list:
    """Drop or recenter flower boxes that are likely face/body false positives.

    Goal:
    - In Custom4 mode, avoid large 'flower' boxes on people.
    - Keep true flowers when a compact petal region exists inside a larger box.
    """
    dets = canonicalize_final_detections(detections)
    if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
        return dets
    if img_w <= 0 or img_h <= 0:
        return dets

    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        if cls_name != 'flower':
            out.append(d)
            continue

        confv = float(d.get('conf', 0.0))
        xy = d.get('xyxy') or []
        if len(xy) < 4:
            continue
        area_ratio, _, _ = _box_metrics(xy, img_w, img_h)
        if area_ratio <= 0:
            continue

        # Keep compact flower boxes even in human scenes; larger boxes must pass skin/petal checks.
        if area_ratio < 0.012 and confv >= 0.55:
            out.append(d)
            continue
        if area_ratio < 0.030 and confv >= 0.48:
            out.append(d)
            continue

        try:
            x1, y1, x2, y2 = map(int, xy[:4])
        except Exception:
            out.append(d)
            continue
        x1 = max(0, min(img_w - 1, x1))
        x2 = max(0, min(img_w, x2))
        y1 = max(0, min(img_h - 1, y1))
        y2 = max(0, min(img_h, y2))
        if x2 <= x1 + 6 or y2 <= y1 + 6:
            out.append(d)
            continue

        roi = image_np[y1:y2, x1:x2]
        if roi.size == 0:
            out.append(d)
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        hch = hsv[:, :, 0]
        sch = hsv[:, :, 1]
        vch = hsv[:, :, 2]
        ycc = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
        ych = ycc[:, :, 0]
        cr = ycc[:, :, 1]
        cb = ycc[:, :, 2]

        # Approximate skin mask in YCrCb + HSV.
        skin_mask = (
            (ych >= 40)
            & (cr >= 135)
            & (cr <= 180)
            & (cb >= 85)
            & (cb <= 135)
            & (hch <= 25)
            & (sch >= 25)
        )

        non_green = ((hch < 35) | (hch > 95))
        colorful = (sch >= 55) & (vch >= 40)
        bright_white = (vch >= 170) & (sch <= 55)
        petal_like = (non_green & colorful) | bright_white

        roi_area = float(max(1, roi.shape[0] * roi.shape[1]))
        skin_ratio = float(np.count_nonzero(skin_mask)) / roi_area
        petal_ratio = float(np.count_nonzero(petal_like)) / roi_area
        ch, cw = roi.shape[:2]
        cx1 = int(max(0, cw * 0.25))
        cx2 = int(min(cw, cw * 0.75))
        cy1 = int(max(0, ch * 0.25))
        cy2 = int(min(ch, ch * 0.75))
        if cx2 > cx1 and cy2 > cy1:
            center_mask = petal_like[cy1:cy2, cx1:cx2]
            center_area = float(max(1, center_mask.shape[0] * center_mask.shape[1]))
            center_petal_ratio = float(np.count_nonzero(center_mask)) / center_area
        else:
            center_petal_ratio = petal_ratio

        # If skin dominates but there is a compact petal region, recenter to that region.
        if skin_ratio >= 0.28 and petal_ratio >= 0.015:
            try:
                cc = cv2.connectedComponentsWithStats(petal_like.astype(np.uint8), connectivity=8)
                _, _, stats, _ = cc
                if stats is not None and len(stats) > 1:
                    stats_fg = stats[1:]
                    best_idx = int(np.argmax(stats_fg[:, cv2.CC_STAT_AREA]))
                    comp = stats_fg[best_idx]
                    comp_area = int(comp[cv2.CC_STAT_AREA])
                    if comp_area >= int(0.003 * roi_area):
                        rx = int(comp[cv2.CC_STAT_LEFT])
                        ry = int(comp[cv2.CC_STAT_TOP])
                        rw = int(comp[cv2.CC_STAT_WIDTH])
                        rh = int(comp[cv2.CC_STAT_HEIGHT])
                        pad_x = max(4, int(rw * 0.25))
                        pad_y = max(4, int(rh * 0.25))
                        nx1 = max(0, x1 + rx - pad_x)
                        ny1 = max(0, y1 + ry - pad_y)
                        nx2 = min(img_w, x1 + rx + rw + pad_x)
                        ny2 = min(img_h, y1 + ry + rh + pad_y)
                        new_area, _, _ = _box_metrics([nx1, ny1, nx2, ny2], img_w, img_h)
                        if new_area > 0 and new_area <= (area_ratio * 0.75):
                            d2 = dict(d)
                            d2['xyxy'] = [float(nx1), float(ny1), float(nx2), float(ny2)]
                            out.append(d2)
                            continue
            except Exception:
                pass

        # Hard reject for face/body-dominant boxes (stricter than before).
        if area_ratio >= 0.06 and skin_ratio >= 0.08 and center_petal_ratio < 0.10 and confv < 0.84:
            continue
        if area_ratio >= 0.10 and skin_ratio >= 0.06 and petal_ratio < 0.14 and confv < 0.88:
            continue
        if area_ratio >= 0.20 and center_petal_ratio < 0.12 and confv < 0.92:
            continue
        if skin_ratio >= 0.22 and petal_ratio < 0.18 and confv < 0.80:
            continue
        if skin_ratio >= 0.16 and petal_ratio < 0.12 and area_ratio >= 0.03 and confv < 0.70:
            continue
        if skin_ratio >= 0.35 and area_ratio >= 0.20 and confv < 0.85:
            continue
        if skin_ratio >= 0.35 and area_ratio >= 0.09 and confv < 0.90:
            continue

        # If mostly skin and weak petal evidence -> drop.
        if skin_ratio >= 0.30 and petal_ratio < 0.08 and confv < 0.92:
            continue

        out.append(d)

    return out


def suppress_flower_boxes_on_faces(image_np: np.ndarray, detections: list, img_w: int, img_h: int) -> list:
    """Drop flower boxes that strongly overlap detected human faces."""
    dets = canonicalize_final_detections(detections)
    if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
        return dets
    if img_w <= 0 or img_h <= 0:
        return dets

    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
            else:
                _FACE_CASCADE = False
        except Exception:
            _FACE_CASCADE = False
    if _FACE_CASCADE is False:
        return dets

    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
        )
    except Exception:
        return dets
    if faces is None or len(faces) == 0:
        return dets

    face_boxes = []
    for (fx, fy, fw, fh) in faces:
        x1 = float(max(0, fx))
        y1 = float(max(0, fy))
        x2 = float(min(img_w, fx + fw))
        y2 = float(min(img_h, fy + fh))
        if x2 <= x1 or y2 <= y1:
            continue
        face_boxes.append([x1, y1, x2, y2])
    if not face_boxes:
        return dets

    face_count = len(face_boxes)
    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        if cls_name != 'flower':
            out.append(d)
            continue
        confv = float(d.get('conf', 0.0))
        xy = d.get('xyxy') or []
        if len(xy) < 4:
            continue
        area_ratio, _, _ = _box_metrics(xy, img_w, img_h)
        if area_ratio <= 0:
            continue
        if area_ratio <= 0.03 and confv >= 0.48:
            out.append(d)
            continue
        if area_ratio <= 0.05 and confv >= 0.60:
            out.append(d)
            continue

        try:
            x1, y1, x2, y2 = map(float, xy[:4])
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
        except Exception:
            out.append(d)
            continue

        max_iou = 0.0
        center_on_face = False
        if face_count >= 2:
            # Group selfie/person scenes: flower predictions are often face-like hallucinations.
            if confv < 0.88:
                continue
        for fxy in face_boxes:
            max_iou = max(max_iou, _box_iou(xy, fxy))
            fx1, fy1, fx2, fy2 = fxy
            if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                center_on_face = True

        if max_iou >= 0.16 and area_ratio > 0.03 and confv < 0.90:
            continue
        if face_count >= 2 and max_iou >= 0.04 and area_ratio > 0.03 and confv < 0.95:
            continue
        if center_on_face and area_ratio >= 0.07 and confv < 0.95:
            continue
        out.append(d)

    return out


def suppress_dumbbell_on_human_faces(image_np: np.ndarray, detections: list, img_w: int, img_h: int) -> list:
    """Drop likely dumbbell false positives in human-face scenes (custom-only safety)."""
    dets = canonicalize_final_detections(detections)
    if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
        return dets
    if img_w <= 0 or img_h <= 0:
        return dets

    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
            else:
                _FACE_CASCADE = False
        except Exception:
            _FACE_CASCADE = False
    if _FACE_CASCADE is False:
        return dets

    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    except Exception:
        return dets

    face_count = int(len(faces)) if faces is not None else 0
    if face_count <= 0:
        return dets

    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        if cls_name != 'dumbbell':
            out.append(d)
            continue
        confv = float(d.get('conf', 0.0))
        area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if face_count >= 2 and confv < 0.65:
            continue
        if face_count >= 1 and area_ratio >= 0.03 and confv < 0.45:
            continue
        out.append(d)
    return out


def recenter_flower_boxes_by_visual_evidence(image_np: np.ndarray, detections: list) -> list:
    dets = canonicalize_final_detections(detections)
    if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
        return dets
    h, w = image_np.shape[:2]
    if h <= 0 or w <= 0:
        return dets
    flower_n = sum(1 for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower')
    dense_mode = flower_n >= 10
    sparse_large_mode = flower_n <= 4
    mid_mode = (flower_n >= 5 and flower_n <= 9)
    # Recenter in dense scenes and also in low/mid-count scenes where broad boxes are common.
    if not dense_mode and not sparse_large_mode and not mid_mode:
        return dets

    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        if cls_name != 'flower':
            out.append(d)
            continue
        area_ratio, _, _ = _box_metrics(d.get('xyxy'), w, h)
        confv = float(d.get('conf', 0.0))
        if sparse_large_mode and area_ratio < 0.06:
            out.append(d)
            continue
        if mid_mode and area_ratio < 0.08:
            out.append(d)
            continue
        xy = d.get('xyxy') or []
        if len(xy) < 4:
            out.append(d)
            continue
        try:
            x1, y1, x2, y2 = map(int, xy[:4])
        except Exception:
            out.append(d)
            continue
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 + 4 or y2 <= y1 + 6:
            out.append(d)
            continue

        roi = image_np[y1:y2, x1:x2]
        if roi.size == 0:
            out.append(d)
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        hch = hsv[:, :, 0]
        sch = hsv[:, :, 1]
        vch = hsv[:, :, 2]
        ycc = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
        ych = ycc[:, :, 0]
        cr = ycc[:, :, 1]
        cb = ycc[:, :, 2]
        skin_mask = (
            (ych >= 40)
            & (cr >= 135)
            & (cr <= 180)
            & (cb >= 85)
            & (cb <= 135)
            & (hch <= 25)
            & (sch >= 25)
        )
        non_green = ((hch < 35) | (hch > 95))
        colorful = (sch >= 55) & (vch >= 40)
        bright_white = (vch >= 170) & (sch <= 55)
        petal_like = ((non_green & colorful) | bright_white) & (~skin_mask)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(petal_like.astype(np.uint8), connectivity=8)
        if num_labels <= 1 or stats is None:
            out.append(d)
            continue

        roi_area = int(max(1, roi.shape[0] * roi.shape[1]))
        min_comp_area = int(max(24, roi_area * (0.0015 if dense_mode else 0.0025)))
        stats_fg = stats[1:]
        candidates = []
        for comp in stats_fg:
            comp_area = int(comp[cv2.CC_STAT_AREA])
            if comp_area < min_comp_area:
                continue
            candidates.append(comp)
        if not candidates:
            out.append(d)
            continue

        if dense_mode:
            # In dense scenes, choose the largest stable petal component.
            best = max(candidates, key=lambda c: int(c[cv2.CC_STAT_AREA]))
        elif mid_mode:
            # In mid scenes, favor petal component near center while keeping enough area.
            cx_roi = 0.5 * roi.shape[1]
            cy_roi = 0.5 * roi.shape[0]
            def _score_mid(comp):
                rx = float(comp[cv2.CC_STAT_LEFT])
                ry = float(comp[cv2.CC_STAT_TOP])
                rw = float(comp[cv2.CC_STAT_WIDTH])
                rh = float(comp[cv2.CC_STAT_HEIGHT])
                rcx = rx + 0.5 * rw
                rcy = ry + 0.5 * rh
                dist = math.hypot(rcx - cx_roi, rcy - cy_roi)
                area = float(comp[cv2.CC_STAT_AREA])
                return area - (dist * 0.55)
            best = max(candidates, key=_score_mid)
        else:
            # In sparse-large scenes, prefer component near ROI center to avoid face/background drift.
            cx_roi = 0.5 * roi.shape[1]
            cy_roi = 0.5 * roi.shape[0]
            def _score(comp):
                rx = float(comp[cv2.CC_STAT_LEFT])
                ry = float(comp[cv2.CC_STAT_TOP])
                rw = float(comp[cv2.CC_STAT_WIDTH])
                rh = float(comp[cv2.CC_STAT_HEIGHT])
                rcx = rx + 0.5 * rw
                rcy = ry + 0.5 * rh
                dist = math.hypot(rcx - cx_roi, rcy - cy_roi)
                area = float(comp[cv2.CC_STAT_AREA])
                return area - (dist * 0.75)
            best = max(candidates, key=_score)

        px1 = int(best[cv2.CC_STAT_LEFT])
        py1 = int(best[cv2.CC_STAT_TOP])
        px2 = int(best[cv2.CC_STAT_LEFT] + best[cv2.CC_STAT_WIDTH])
        py2 = int(best[cv2.CC_STAT_TOP] + best[cv2.CC_STAT_HEIGHT])
        spread_x = float(px2 - px1) / float(max(1, roi.shape[1]))
        spread_y = float(py2 - py1) / float(max(1, roi.shape[0]))
        spread_floor = 0.18 if dense_mode else 0.12
        if spread_x < spread_floor or spread_y < spread_floor:
            out.append(d)
            continue

        pad_ratio = 0.18 if dense_mode else 0.25
        pad_x = max(4, int((px2 - px1) * pad_ratio))
        pad_y = max(4, int((py2 - py1) * pad_ratio))
        nx1 = max(0, x1 + px1 - pad_x)
        ny1 = max(0, y1 + py1 - pad_y)
        nx2 = min(w, x1 + px2 + pad_x)
        ny2 = min(h, y1 + py2 + pad_y)
        if nx2 <= nx1 or ny2 <= ny1:
            out.append(d)
            continue
        new_area, _, _ = _box_metrics([nx1, ny1, nx2, ny2], w, h)
        # Reject broad boxes that fail to tighten in low/mid-count flower scenes.
        if (sparse_large_mode or mid_mode) and area_ratio >= 0.08:
            if new_area >= (area_ratio * 0.95):
                if area_ratio >= 0.14 and confv < 0.80:
                    continue
                out.append(d)
                continue
        d2 = dict(d)
        d2['xyxy'] = [float(nx1), float(ny1), float(nx2), float(ny2)]
        out.append(d2)

    return out


def refine_flower_boxes_with_visual_evidence(image_np: np.ndarray, detections: list, img_w: int, img_h: int) -> list:
    dets = canonicalize_final_detections(detections)
    if not dets:
        return []
    dets = suppress_face_like_flower_boxes(image_np, dets, img_w=img_w, img_h=img_h)
    dets = suppress_flower_boxes_on_faces(image_np, dets, img_w=img_w, img_h=img_h)
    dets = suppress_flower_on_fruit_confusions(dets, img_w=img_w, img_h=img_h, image_np=image_np)
    dets = prune_flower_boxes_by_visual_evidence(image_np, dets)
    dets = recenter_flower_boxes_by_visual_evidence(image_np, dets)
    flower_n = sum(1 for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower')
    if flower_n >= 12:
        nms_iou = 0.85
    elif flower_n >= 6:
        nms_iou = 0.75
    else:
        nms_iou = 0.55
    dets = dedup_detections_by_class_nms_classwise(dets, default_iou=nms_iou)
    return dets


def build_counts_from_detections(final_detections: list, min_conf: float | None = None) -> dict:
    counts = {}
    for d in final_detections or []:
        try:
            confv = float(d.get('conf', 0.0))
        except Exception:
            confv = 0.0
        if min_conf is not None and confv < float(min_conf):
            continue
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts


def summarize_detection_stage(detections: list) -> dict:
    dets = canonicalize_final_detections(detections)
    counts = build_counts_from_detections(dets)
    avg_conf = (sum(float(d.get('conf', 0.0)) for d in dets) / len(dets)) if dets else 0.0
    top_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:6]
    return {
        'total': int(len(dets)),
        'avg_conf': round(float(avg_conf), 4),
        'top': [{'class': str(k), 'count': int(v)} for k, v in top_items],
    }


def extract_final_detections_from_raw(raw: dict) -> list:
    if not isinstance(raw, dict):
        return []
    if isinstance(raw.get('detections'), list):
        return canonicalize_final_detections(raw.get('detections') or [])
    if isinstance(raw.get('frames'), list):
        frames = [f for f in (raw.get('frames') or []) if isinstance(f, dict)]
        if not frames:
            return []

        # Video rule: count by one frame only, never accumulate across frames.
        frame_ids = [int(f.get('frame', 0)) for f in frames]
        target_frame = min(frame_ids) if frame_ids else 0
        frame_dets = []
        for f in frames:
            if int(f.get('frame', 0)) != int(target_frame):
                continue
            frame_dets.append({
                'class': f.get('class', ''),
                'conf': f.get('conf', 0.0),
                'xyxy': f.get('xyxy'),
            })
        return canonicalize_final_detections(frame_dets)
    return []


def verify_and_reduce_detections(
    final_detections: list,
    img_w: int,
    img_h: int,
    base_conf: float = 0.25,
    strict_coco: bool = False,
    scene_rule_debug: dict | None = None,
) -> list:
    custom_classes = get_custom_class_names()
    dets = canonicalize_final_detections(final_detections)
    scene_classes = {normalize_class_name(d.get('class', '')) or 'unknown' for d in dets}
    traffic_scene = any(c in {'person', 'car', 'motorcycle', 'bicycle', 'bus', 'truck'} for c in scene_classes)
    has_other_custom = any(c in {'flower', 'fruit', 'tree', 'dumbbell'} for c in scene_classes)
    filtered = []

    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        confv = float(d.get('conf', 0.0))
        area_ratio, aspect, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)

        is_custom = cls_name in custom_classes
        if is_custom:
            min_conf = max(0.16, float(base_conf) * 0.70)
            if cls_name == 'tree':
                # Allow lower confidence for custom tree to improve recall in real-world scenes
                min_conf = min(min_conf, max(0.08, float(base_conf) * 0.35))
            if cls_name == 'dumbbell':
                min_conf = min(min_conf, 0.30)
            if cls_name == 'flower':
                # Keep flower stricter to reduce false positives on face/background regions.
                min_conf = max(0.30, float(base_conf) * 0.95)
        else:
            min_conf = max(0.32, float(base_conf) + (0.18 if strict_coco else 0.08))

        if confv < min_conf:
            continue
        if area_ratio <= 0:
            continue
        if cls_name == 'tree':
            # Relax tree gates slightly, especially for custom-sourced detections.
            src = str(d.get('_source_model', '') or '').lower()
            is_custom_src = 'custom' in src
            tree_conf_floor = float(TREE_MIN_CONF)
            tree_area_floor = float(TREE_MIN_AREA)
            if is_custom_src:
                # More permissive for custom tree detections
                tree_conf_floor = max(0.08, float(TREE_MIN_CONF) * 0.55)
                tree_area_floor = max(0.00004, float(TREE_MIN_AREA) * 0.6)

            if confv < tree_conf_floor:
                if scene_rule_debug is not None:
                    scene_rule_debug.setdefault('drops_by_rule', {})
                    scene_rule_debug['drops_by_rule'][f'tree|conf_lt_{tree_conf_floor}'] = scene_rule_debug['drops_by_rule'].get(f'tree|conf_lt_{tree_conf_floor}', 0) + 1
                continue
            if area_ratio < tree_area_floor:
                if scene_rule_debug is not None:
                    scene_rule_debug.setdefault('drops_by_rule', {})
                    scene_rule_debug['drops_by_rule'][f'tree|area_lt_{tree_area_floor}'] = scene_rule_debug['drops_by_rule'].get(f'tree|area_lt_{tree_area_floor}', 0) + 1
                continue
        if cls_name == 'dumbbell':
            # Dedicated dumbbell gate: permissive recall floor.
            if confv < DUMBBELL_MIN_CONF:
                if scene_rule_debug is not None:
                    scene_rule_debug.setdefault('drops_by_rule', {})
                    scene_rule_debug['drops_by_rule'][f'dumbbell|conf_lt_{DUMBBELL_MIN_CONF}'] = scene_rule_debug['drops_by_rule'].get(f'dumbbell|conf_lt_{DUMBBELL_MIN_CONF}', 0) + 1
                continue
            if area_ratio < DUMBBELL_MIN_AREA:
                if scene_rule_debug is not None:
                    scene_rule_debug.setdefault('drops_by_rule', {})
                    scene_rule_debug['drops_by_rule'][f'dumbbell|area_lt_{DUMBBELL_MIN_AREA}'] = scene_rule_debug['drops_by_rule'].get(f'dumbbell|area_lt_{DUMBBELL_MIN_AREA}', 0) + 1
                continue
            # Guardrail: drop low-confidence, oversized dumbbell boxes (common false positives on vehicles).
            if area_ratio > 0.20 and confv < max(0.45, float(base_conf) + 0.15):
                continue
            # Guardrail: unreasonable aspect ratios for dumbbells (avoid square/ultra-long boxes).
            if (aspect < 1.10 or aspect > 6.0) and confv < 0.60:
                continue
            if strict_coco and not has_other_custom and traffic_scene and confv < max(0.60, float(base_conf) + 0.20):
                continue
            if (not strict_coco) and (not has_other_custom) and confv < max(0.32, float(base_conf) + 0.08):
                continue
        if cls_name == 'flower':
            # Stronger flower gating: avoid ghost boxes while preserving true tiny flowers.
            tiny_floor = max(0.30, float(base_conf) * 0.90)
            if area_ratio < 0.00020 and confv < tiny_floor:
                continue
            if area_ratio < 0.00012 and confv < 0.42:
                continue
            if area_ratio > 0.10 and confv < 0.45:
                if scene_rule_debug is not None:
                    scene_rule_debug.setdefault('drops_by_rule', {})
                    scene_rule_debug['drops_by_rule']['flower|large_low_conf'] = scene_rule_debug['drops_by_rule'].get('flower|large_low_conf', 0) + 1
                continue
        else:
            # For non-flower classes, be less aggressive about dropping tiny boxes
            # to avoid missing small objects; lower the required confidence.
            tiny_floor = 0.50
            if cls_name in {'tree', 'dumbbell'}:
                tiny_floor = 0.40
            # allow fruits to be kept at a lower tiny-floor to avoid losing small fruit boxes
            if cls_name == 'fruit' or cls_name in SPECIFIC_FRUIT_CLASSES:
                tiny_floor = 0.30
            if area_ratio < 0.00035 and confv < tiny_floor:
                continue
        # keep near-full boxes for custom classes with moderate confidence;
        # only apply stricter rule to non-custom classes to reduce COCO noise.
        if near_full:
            # Only drop near-full (whole-image) boxes when they are weak.
            # Keep them if they meet stronger class/source-specific confidence floors.
            if is_custom and confv < 0.68:
                continue
            if (not is_custom) and confv < 0.86:
                continue
            # otherwise keep the near-full box (do not unconditional continue)
        if aspect > 10.0 and confv < 0.90:
            continue

        d2 = dict(d)
        d2['class'] = cls_name
        filtered.append(d2)

    # Class-aware NMS to reduce same-class duplicates
    filtered = dedup_detections_by_class_nms_classwise(filtered, default_iou=0.50)

    # Source-aware cross-class suppression for sensitive confusion pairs
    filtered = resolve_cross_class_overlaps_with_priority(
        filtered,
        iou_thresh=0.60,
        conf_gap=0.06,
    )

    # Defensive guard: ensure downstream code receives a list (avoid None crashes)
    if filtered is None:
        filtered = []

    # Cross-class suppression: if 2 classes heavily overlap, keep stronger one
    ordered = sorted(filtered, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    kept = []
    for d in ordered:
        drop = False
        for k in kept:
            if str(d.get('class', '')) == str(k.get('class', '')):
                continue
            iou = _box_iou(d.get('xyxy'), k.get('xyxy'))
            pair = {str(d.get('class', '')), str(k.get('class', ''))}
            if pair == {'flower', 'dumbbell'} and iou >= 0.35:
                d_conf = float(d.get('conf', 0.0))
                k_conf = float(k.get('conf', 0.0))
                d_area, d_aspect, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
                k_area, k_aspect, _ = _box_metrics(k.get('xyxy'), img_w, img_h)
                d_cls = str(d.get('class', ''))
                k_cls = str(k.get('class', ''))
                d_is_db = (d_cls == 'dumbbell')
                k_is_db = (k_cls == 'dumbbell')
                # Prefer higher confidence; then use geometry as tie-breaker.
                if d_is_db and d_conf + 0.06 < k_conf:
                    drop = True
                    break
                if (not d_is_db) and k_is_db and k_conf + 0.06 >= d_conf:
                    drop = True
                    break
                if d_is_db and (d_area < 0.0006 or d_aspect < 1.08):
                    drop = True
                    break
                if (not d_is_db) and k_is_db and (k_area < 0.0006 or k_aspect < 1.08):
                    continue
            if iou >= 0.78 and float(d.get('conf', 0.0)) <= float(k.get('conf', 0.0)) + 0.10:
                drop = True
                break
        if not drop:
            kept.append(d)

    # In fruit-heavy scenes, suppress weak tree boxes overlapping fruit clusters.
    fruit_refs = [
        d for d in kept
        if ((normalize_class_name(d.get('class', '')) or '') == 'fruit' or (normalize_class_name(d.get('class', '')) or '') in SPECIFIC_FRUIT_CLASSES)
        and float(d.get('conf', 0.0)) >= 0.45
    ]
    if len(fruit_refs) >= 2:
        reduced = []
        for d in kept:
            cls_name = normalize_class_name(d.get('class', '')) or ''
            confv = float(d.get('conf', 0.0))
            if cls_name == 'tree' and confv < 0.65:
                overlap_fruit = any(_box_iou(d.get('xyxy'), f.get('xyxy')) >= 0.10 for f in fruit_refs)
                if overlap_fruit:
                    if scene_rule_debug is not None:
                        scene_rule_debug.setdefault('drops_by_rule', {})
                        scene_rule_debug['drops_by_rule']['tree|overlap_fruit_scene_low_conf'] = scene_rule_debug['drops_by_rule'].get('tree|overlap_fruit_scene_low_conf', 0) + 1
                    continue
            reduced.append(d)
        kept = reduced

    has_strong_dumbbell = any(
        (normalize_class_name(d.get('class', '')) or '') == 'dumbbell'
        and float(d.get('conf', 0.0)) >= 0.30
        for d in kept
    )
    fruit_heavy_scene = len(fruit_refs) >= 2

    scene_rescue = [
        dict(d)
        for d in dets
        if (
            (normalize_class_name(d.get('class', '')) or '') in {'tree', 'dumbbell'}
            and not (
                (normalize_class_name(d.get('class', '')) or '') == 'tree'
                and fruit_heavy_scene
                and not has_strong_dumbbell
            )
        )
        and float(d.get('conf', 0.0)) >= 0.22
    ]

    kept = _apply_scene_context_rules(
        kept,
        img_w=img_w,
        img_h=img_h,
        base_conf=float(base_conf),
        debug_drop_counts=scene_rule_debug,
    )
    kept = suppress_flower_cross_class_confusions(
        kept,
        img_w=img_w,
        img_h=img_h,
        base_conf=float(base_conf),
    )
    tree_now = int(sum(1 for d in kept if (normalize_class_name(d.get('class', '')) or '') == 'tree'))
    dumbbell_now = int(sum(1 for d in kept if (normalize_class_name(d.get('class', '')) or '') == 'dumbbell'))
    if (tree_now == 0 or dumbbell_now == 0) and scene_rescue:
        add_back = [
            d for d in scene_rescue
            if ((normalize_class_name(d.get('class', '')) or '') == 'tree' and tree_now == 0)
            or ((normalize_class_name(d.get('class', '')) or '') == 'dumbbell' and dumbbell_now == 0)
        ]
        if add_back:
            kept = dedup_detections_by_class_nms_classwise(kept + add_back, default_iou=0.60)

    # Guardrail: prevent runaway counts from noisy classes
    class_caps = {
        'flower': 180,
        'fruit': 80,
        'tree': 60,
        'dumbbell': 40,
        'unknown': 60,
    }
    out = []
    per_class_count = {}
    for d in kept:
        c = str(d.get('class', 'unknown')) or 'unknown'
        n = per_class_count.get(c, 0)
        if n >= int(class_caps.get(c, 60)):
            continue
        per_class_count[c] = n + 1
        out.append(d)
    return out


def _apply_scene_context_rules(
    detections: list,
    img_w: int,
    img_h: int,
    base_conf: float,
    debug_drop_counts: dict | None = None,
) -> list:
    dets = canonicalize_final_detections(detections)
    if not dets:
        return []

    counts = {}
    top_conf = {}
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        confv = float(d.get('conf', 0.0))
        counts[cls_name] = counts.get(cls_name, 0) + 1
        top_conf[cls_name] = max(float(top_conf.get(cls_name, 0.0)), confv)

    def _is_custom_src(d):
        try:
            src = str(d.get('_source_model', '') or '').lower().strip()
            return 'custom' in src
        except Exception:
            return False

    custom_flower_present = any(
        (normalize_class_name(d.get('class', '')) or '') == 'flower' and _is_custom_src(d)
        for d in dets
    )

    flower_n = int(counts.get('flower', 0))
    fruit_n = int(counts.get('fruit', 0))
    tree_n = int(counts.get('tree', 0))
    dumbbell_n = int(counts.get('dumbbell', 0))
    other_n = max(0, len(dets) - flower_n)

    # traffic scene detection (cars, buses, people, etc.)
    traffic_scene = any(k in counts for k in ('person', 'car', 'motorcycle', 'bicycle', 'bus', 'truck'))

    flower_heavy = (flower_n >= 4 and flower_n >= (other_n * 2 + 1))
    tree_heavy = (tree_n >= 3)
    non_flower_heavy = (other_n >= 3 and flower_n <= 1)
    dumbbell_heavy = (
        dumbbell_n >= 1
        and float(top_conf.get('dumbbell', 0.0)) >= max(0.85, float(base_conf) + 0.35)
        and flower_n <= 2
    )

    dumbbell_refs = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'dumbbell']

    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        confv = float(d.get('conf', 0.0))
        area_ratio, aspect, _ = _box_metrics(d.get('xyxy'), img_w, img_h)

        if flower_heavy:
            if cls_name == 'dumbbell':
                if confv < max(0.58, float(base_conf) + 0.12):
                    continue
                if area_ratio < max(DUMBBELL_MIN_AREA * 8.0, 0.0006):
                    continue
                if aspect < 1.15 or aspect > 10.0:
                    continue
            if cls_name == 'fruit' and confv < 0.78:
                continue
            if cls_name == 'vase' and confv < 0.92:
                continue

        if dumbbell_heavy and cls_name == 'fruit':
            overlap_dumbbell = any(
                _box_iou(d.get('xyxy'), db.get('xyxy')) >= 0.20
                and confv <= float(db.get('conf', 0.0)) + 0.15
                for db in dumbbell_refs
            )
            if overlap_dumbbell or confv < 0.70:
                continue

        # In non-flower scenes with many other objects, suppress weak flower hallucinations.
        if cls_name == 'flower' and non_flower_heavy:
            if confv < 0.80:
                continue
            if area_ratio < 0.004:
                continue

        # In flower-dominant scenes, suppress weak tree hallucinations.
        # Relax thresholds when the scene is tree-heavy.
        if cls_name == 'tree' and flower_n > 2:
            # Relax tree suppression in flower-heavy scenes while still guarding noise.
            # Lower confidence/area floors to avoid losing valid trees, especially when
            # custom model provides tree detections.
            custom_present = 'tree' in get_custom_class_names()
            if tree_heavy:
                tree_conf_floor = max(0.48, float(base_conf) + 0.08)
                tree_area_floor = 0.005
                tree_repeat_conf = 0.55
            else:
                tree_conf_floor = max(0.50, float(base_conf) + 0.10)
                tree_area_floor = 0.01
                tree_repeat_conf = 0.60

            # If custom model includes 'tree', be even more permissive to preserve custom detections
            if custom_present:
                tree_conf_floor = min(tree_conf_floor, float(base_conf) + 0.06)
                tree_area_floor = max(0.0008, tree_area_floor * 0.6)

            if confv < tree_conf_floor:
                if debug_drop_counts is not None:
                    debug_drop_counts['drop_tree_flower_scene_low_conf'] = int(debug_drop_counts.get('drop_tree_flower_scene_low_conf', 0)) + 1
                continue
            if area_ratio < tree_area_floor:
                if debug_drop_counts is not None:
                    debug_drop_counts['drop_tree_flower_scene_small_area'] = int(debug_drop_counts.get('drop_tree_flower_scene_small_area', 0)) + 1
                continue
            if tree_n >= 1 and confv < tree_repeat_conf:
                if debug_drop_counts is not None:
                    debug_drop_counts['drop_tree_flower_scene_repeat_low_conf'] = int(debug_drop_counts.get('drop_tree_flower_scene_repeat_low_conf', 0)) + 1
                continue

        # In flower-dominant scenes, suppress only very weak fruit confusion.
        if cls_name == 'fruit' and flower_n >= 2:
            if confv < 0.62 and area_ratio < 0.006:
                continue

        # If scene is fruit-heavy and flower is scarce, suppress weak flower boxes
        if cls_name == 'flower' and fruit_n >= 3 and flower_n <= 1:
            if confv < 0.90 or area_ratio < 0.006:
                continue

        # In traffic scenes, flowers are unlikely; be very strict
        if cls_name == 'flower' and traffic_scene:
            if custom_flower_present:
                if confv < 0.50 and area_ratio < 0.002:
                    continue
            else:
                if confv < 0.92 or area_ratio < 0.006:
                    continue

        out.append(d)
    return out


def suppress_flower_cross_class_confusions(detections: list, img_w: int, img_h: int, base_conf: float) -> list:
    """Suppress non-flower hallucinations that overlap flower boxes.

    This is applied late to reduce class confusion in flower-dominant scenes.
    """
    dets = canonicalize_final_detections(detections)
    if not dets:
        return []

    flowers = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        if cls_name != 'flower':
            continue
        confv = float(d.get('conf', 0.0))
        area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if area_ratio <= 0:
            continue
        if confv < max(0.22, float(base_conf) * 0.65):
            continue
        flowers.append(d)

    if not flowers:
        return dets

    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        if cls_name == 'flower':
            out.append(d)
            continue

        if cls_name not in {'fruit', 'tree', 'dumbbell', 'vase'}:
            out.append(d)
            continue

        confv = float(d.get('conf', 0.0))
        area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        best_iou = 0.0
        best_flower_conf = 0.0
        for f in flowers:
            iou = _box_iou(d.get('xyxy'), f.get('xyxy'))
            if iou > best_iou:
                best_iou = iou
                best_flower_conf = float(f.get('conf', 0.0))

        # General overlap conflict rule.
        if best_iou >= 0.25 and confv <= best_flower_conf + 0.12:
            if cls_name in {'tree', 'dumbbell'} and confv >= max(0.55, float(base_conf) + 0.10):
                pass
            else:
                continue

        # Class-specific stricter suppression in flower-heavy scenes.
        if cls_name == 'fruit' and best_iou >= 0.20 and confv < 0.88:
            continue
        if cls_name == 'tree' and best_iou >= 0.22 and confv < max(0.70, float(base_conf) + 0.20):
            continue
        if cls_name == 'dumbbell' and best_iou >= 0.22 and confv < max(0.58, float(base_conf) + 0.12):
            continue
        if cls_name == 'vase' and best_iou >= 0.20 and confv < 0.92:
            continue

        # Large non-flower box overlapping flowers is likely a scene-scan artifact.
        if area_ratio >= 0.45 and best_iou >= 0.08:
            continue

        out.append(d)
    return out


def annotate_image(img: np.ndarray, results, names_map=None, conf_thresh=0.25, totals: dict = None):
    """Draw detected boxes and optionally overlay total counts.

    Args:
        img: source RGB image (numpy array).
        results: model results iterable (each result has `boxes`).
        names_map: optional mapping from class id to name.
        conf_thresh: confidence threshold for drawing boxes.
        totals: optional dict of counts per class to overlay total count.
    Returns:
        Annotated image copy.
    """
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
            display_label = _ascii_safe_label(display_label)
            # draw box + readable label background
            color = _color_for_label(str(label_norm or label or 'unknown'))
            _draw_box_with_label(img, (x1, y1, x2, y2), f"{display_label} {conf:.2f}", color)

    # Draw total counts overlay if provided
    try:
        if totals and isinstance(totals, dict):
            total = sum(int(v) for v in totals.values())
            text = f"Total: {total}"
            # draw background rectangle for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.9
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            pad = 8
            x0, y0 = 10, 10
            cv2.rectangle(img, (x0, y0), (x0 + tw + pad, y0 + th + pad), (0, 0, 0), -1)
            cv2.putText(img, text, (x0 + 4, y0 + th + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    except Exception:
        pass

    return img


def run_detection_on_image(model, image_np, conf: float, imgsz: int = 640, use_fp16: bool = True, nms_iou: float = None, max_det: int = None):
    effective_conf = max(0.10, float(conf))
    predict_kwargs = dict(source=image_np, conf=effective_conf, imgsz=imgsz)
    # use half precision if requested and CUDA available
    if use_fp16 and (torch is not None and torch.cuda.is_available()):
        predict_kwargs['half'] = True
    if torch is not None and torch.cuda.is_available():
        predict_kwargs['device'] = 'cuda'
    try:
        # Use configured NMS IoU or fallback to 0.5 (user requested lower IoU)
        if nms_iou is None:
            nms_iou = float(st.session_state.get('nms_iou', 0.5))
        nms_iou = max(0.45, min(0.80, float(nms_iou)))
        if max_det is None:
            max_det = int(st.session_state.get('max_det', 100))
        max_det = max(50, min(260, int(max_det)))
        predict_kwargs['iou'] = float(nms_iou)
        predict_kwargs['max_det'] = int(max_det)
    except Exception:
        pass
    results = model.predict(**predict_kwargs)
    names = results[0].names if hasattr(results[0], 'names') else {}
    detections = []
    # collect raw detections from results
    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        for box in boxes:
            confv = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
            if confv < effective_conf:
                continue
            cls = int(box.cls[0]) if hasattr(box, 'cls') else None
            name = normalize_class_name(names.get(cls, str(cls)))
            xyxy = None
            try:
                xy = getattr(box, 'xyxy', None)
                xyxy = xy[0].cpu().numpy() if xy is not None else None
            except Exception:
                try:
                    xyxy = np.array(box.xyxy[0])
                except Exception:
                    xyxy = None
            detections.append({
                "class": name,
                "conf": float(confv),
                "xyxy": xyxy.tolist() if xyxy is not None else None,
            })

    # Return lightly-filtered raw detections.
    # NOTE: heavy verification/finalization is handled later by process_uploaded/main.
    dets = canonicalize_final_detections(detections)
    dets = dedup_detections_by_class_nms_classwise(dets, default_iou=0.65)

    # Rebuild counts from current detections (full pipeline may recompute again later).
    counts = build_counts_from_detections(dets)
    ann = image_np.copy() if isinstance(image_np, np.ndarray) else None
    if ann is not None:
        for d in dets:
            xy = d.get('xyxy') or []
            if len(xy) < 4:
                continue
            x1, y1, x2, y2 = map(int, xy[:4])
            label = d.get('class', 'unknown')
            confv = float(d.get('conf', 0.0))
            color = _color_for_label(str(label))
            _draw_box_with_label(ann, (x1, y1, x2, y2), f"{label} {confv:.2f}", color)
        # overlay totals as before
        try:
            if counts:
                total = sum(int(v) for v in counts.values())
                text = f"Total: {total}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.9
                thickness = 2
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                pad = 8
                x0, y0 = 10, 10
                cv2.rectangle(ann, (x0, y0), (x0 + tw + pad, y0 + th + pad), (0, 0, 0), -1)
                cv2.putText(ann, text, (x0 + 4, y0 + th + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        except Exception:
            pass
    return ann, counts, dets


def run_tiled_flower_detections(model, image_np: np.ndarray, base_conf: float, imgsz: int = 640, use_fp16: bool = True):
    """Fallback for dense flower scenes: run inference on overlapping tiles.
    This helps recover multiple instances when full-image inference collapses to one large flower box.
    """
    if image_np is None or image_np.size == 0:
        return []

    h, w = image_np.shape[:2]
    if h < 64 or w < 64:
        return []

    tile_w = min(w, max(160, FAST_TILE_SIZE))
    tile_h = min(h, max(160, FAST_TILE_SIZE))
    stride_x = max(64, min(tile_w, FAST_TILE_STRIDE))
    stride_y = max(64, min(tile_h, FAST_TILE_STRIDE))

    xs = list(range(0, max(1, w - tile_w + 1), stride_x))
    ys = list(range(0, max(1, h - tile_h + 1), stride_y))
    if not xs or xs[-1] != max(0, w - tile_w):
        xs.append(max(0, w - tile_w))
    if not ys or ys[-1] != max(0, h - tile_h):
        ys.append(max(0, h - tile_h))

    # PATCH: reduce flower FP / fruit confusion — use more permissive tiled pass for dense scenes
    # For the user's request, prefer a low tile_conf to maximize recall in tiles,
    # then rely on visual pruning to remove leaf/texture false positives.
    tile_conf = 0.10
    tile_jobs = []
    for y0 in ys:
        for x0 in xs:
            tile = image_np[y0:y0 + tile_h, x0:x0 + tile_w]
            if tile.size == 0:
                continue
            tile_jobs.append((x0, y0, tile))

    out = []
    for batch_start in range(0, len(tile_jobs), FAST_TILE_BATCH_SIZE):
        batch_jobs = tile_jobs[batch_start:batch_start + FAST_TILE_BATCH_SIZE]
        batch_tiles = [job[2] for job in batch_jobs]
        predict_kwargs = dict(
            source=batch_tiles,
            conf=tile_conf,
            imgsz=imgsz,
            batch=min(FAST_TILE_BATCH_SIZE, len(batch_tiles)),
            iou=0.72,
            max_det=512,
        )
        if use_fp16 and (torch is not None and torch.cuda.is_available()):
            predict_kwargs['half'] = True
        if torch is not None and torch.cuda.is_available():
            predict_kwargs['device'] = 'cuda'

        results_batch = model.predict(**predict_kwargs)
        for (x0, y0, _), result in zip(batch_jobs, results_batch):
            names = result.names if hasattr(result, 'names') else {}
            boxes = getattr(result, 'boxes', None)
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
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                area_ratio = (bw * bh) / float(max(1, w * h))
                # filter tiny detections only when confidence very low
                if area_ratio < 0.00006 and confv < 0.30:
                    continue
                if area_ratio > 0.65 and confv < 0.90:
                    continue
                out.append({
                    'class': 'flower',
                    'conf': float(confv),
                    'xyxy': [x1 + x0, y1 + y0, x2 + x0, y2 + y0],
                })

    if not out:
        return []

    # Collapse overlapping low-petal detections: keep highest-conf in tight-overlap groups
    cand = canonicalize_final_detections(out)
    # helper to compute petal_fill_ratio for a detection
    def _petal_fill_for_box(box):
        try:
            x1, y1, x2, y2 = map(int, box.get('xyxy')[:4])
        except Exception:
            return 1.0
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return 1.0
        roi = image_np[y1:y2, x1:x2]
        if roi.size == 0:
            return 1.0
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            hch = hsv[:, :, 0]
            sch = hsv[:, :, 1]
            vch = hsv[:, :, 2]
            non_green = ((hch < 35) | (hch > 95))
            colorful = (sch >= 55) & (vch >= 40)
            bright_white = (vch >= 170) & (sch <= 55)
            petal_like = (non_green & colorful) | bright_white
            ys, xs = np.where(petal_like)
            if len(xs) == 0:
                return 0.0
            px1 = int(xs.min())
            py1 = int(ys.min())
            px2 = int(xs.max()) + 1
            py2 = int(ys.max()) + 1
            pbox_area = max(1, (px2 - px1) * (py2 - py1))
            petal_fill_ratio = float(len(xs)) / float(pbox_area)
            return float(petal_fill_ratio)
        except Exception:
            return 1.0

    kept = []
    ordered = sorted(cand, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    for d in ordered:
        dup = False
        for k in kept:
            if _box_iou(d.get('xyxy'), k.get('xyxy')) >= 0.60:
                # if this box has very low petal evidence, skip it in favor of kept
                pf = _petal_fill_for_box(d)
                if pf < 0.06:
                    dup = True
                    break
                # otherwise keep both; but if d is clearly lower conf and low petal, drop
                if float(d.get('conf', 0.0)) <= float(k.get('conf', 0.0)) - 0.06 and pf < 0.10:
                    dup = True
                    break
        if not dup:
            kept.append(d)

    # final dedup IoU per user request
    dedup_iou = 0.74
    return dedup_detections_by_class_nms_classwise(kept, default_iou=float(dedup_iou))


def refine_custom_detections_with_specialists(image_np: np.ndarray, detections: list, base_conf: float, imgsz: int = 640, use_fp16: bool = True):
    """Refine custom classes using specialist models to reduce false positives and recover misses.
    - Keep high-confidence custom detections directly.
    - Require specialist overlap for lower-confidence custom detections.
    - Add specialist detections not already covered.
    """
    if image_np is None or image_np.size == 0:
        return canonicalize_final_detections(detections)

    if not USE_SPECIALIST_MODELS:
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
    flower_specialist_model = None
    for cls_name in custom_classes:
        path = model_map.get(cls_name)
        if not path or not os.path.exists(path):
            continue
        try:
            specialist_model = load_model(path)
            if cls_name == 'flower':
                flower_specialist_model = specialist_model
                f_conf = float(max(0.12, float(base_conf) * 0.55))
                _, _, dets_a = run_detection_on_image(
                    specialist_model,
                    image_np,
                    conf=f_conf,
                    imgsz=max(int(imgsz), 768),
                    use_fp16=use_fp16,
                    nms_iou=0.78,
                    max_det=260,
                )
                _, _, dets_b = run_detection_on_image(
                    specialist_model,
                    image_np,
                    conf=max(0.10, f_conf - 0.02),
                    imgsz=max(int(imgsz), 1024),
                    use_fp16=use_fp16,
                    nms_iou=0.80,
                    max_det=320,
                )
                tiled = run_tiled_flower_detections(
                    specialist_model,
                    image_np,
                    base_conf=max(0.10, f_conf),
                    imgsz=max(int(imgsz), 960),
                    use_fp16=use_fp16,
                )
                flower_merged = canonicalize_final_detections(dets_a + dets_b + tiled)
                flower_merged = [d for d in flower_merged if (normalize_class_name(d.get('class', '')) or '') == 'flower']
                specialists[cls_name] = dedup_detections_by_class_nms_classwise(flower_merged, default_iou=0.90)
            else:
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

    # Recover misses from specialists (especially flower + dumbbell/tree)
    for cls_name in custom_classes:
        for s in specialists.get(cls_name, []):
            overlap = any(
                (normalize_class_name(r.get('class', '')) == cls_name)
                and _box_iou(r.get('xyxy'), s.get('xyxy')) >= (0.30 if cls_name == 'flower' else 0.40)
                for r in refined
            )
            if not overlap:
                refined.append(s)

    refined = dedup_detections_by_class_nms_classwise(refined, default_iou=0.60)

    # Final flower prioritization and cleanup: enforce 1-flower-1-box as late as possible.
    try:
        h, w = image_np.shape[:2]
    except Exception:
        h, w = 0, 0
    if h > 0 and w > 0:
        if flower_specialist_model is not None:
            # If specialist found richer flower set than merged output, prefer it.
            current_flower = [d for d in refined if (normalize_class_name(d.get('class', '')) or '') == 'flower']
            spec_flower = specialists.get('flower', []) or []
            if len(spec_flower) > len(current_flower):
                non_flower = [d for d in refined if (normalize_class_name(d.get('class', '')) or '') != 'flower']
                refined = non_flower + spec_flower

        refined = suppress_scene_level_boxes(refined, img_w=w, img_h=h)
        refined = enforce_one_object_one_box(refined, img_w=w, img_h=h)
        refined = collapse_sparse_flower_duplicates(refined, img_w=w, img_h=h)
        refined = suppress_flower_cross_class_confusions(refined, img_w=w, img_h=h, base_conf=float(base_conf))

    return refined


def run_detection_on_video(model, video_bytes, conf: float, imgsz: int = 640, use_fp16: bool = True, max_frames=30, sample_rate=5, batch_size: int = 4):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmp.write(video_bytes)
    tmp.flush()
    tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    frame_idx = 0
    first_annotated = None
    preview_counts = {}
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
                predict_kwargs = dict(source=frames_batch, conf=max(float(conf), MIN_COUNT_CONFIDENCE), imgsz=imgsz)
                if use_fp16 and (torch is not None and torch.cuda.is_available()):
                    predict_kwargs['half'] = True
                try:
                    # pass NMS IoU and max_det when available (tunable in UI)
                    predict_kwargs['iou'] = float(st.session_state.get('nms_iou', 0.5))
                    predict_kwargs['max_det'] = int(st.session_state.get('max_det', 100))
                except Exception:
                    pass
                results_batch = model.predict(**predict_kwargs)
                for i, results in enumerate(results_batch):
                    names = results[0].names if hasattr(results[0], 'names') else {}
                    fidx = frames_batch_idx[i]
                    raw_frame_dets = []
                    for r in results:
                        boxes = getattr(r, 'boxes', None)
                        if boxes is None:
                            continue
                        for box in boxes:
                            confv = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                            if confv < MIN_COUNT_CONFIDENCE:
                                continue
                            cls = int(box.cls[0]) if hasattr(box, 'cls') else None
                            name = normalize_class_name(names.get(cls, str(cls)))
                            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else None
                            raw_frame_dets.append({
                                "frame": int(fidx),
                                "class": name,
                                "conf": float(confv),
                                "xyxy": xyxy.tolist() if xyxy is not None else None,
                            })

                    frame_img = frames_batch[i]
                    fh, fw = frame_img.shape[:2] if isinstance(frame_img, np.ndarray) else (0, 0)
                    final_frame_dets = finalize_frame_detections_for_count(
                        raw_frame_dets,
                        img_w=int(fw),
                        img_h=int(fh),
                        min_conf=MIN_COUNT_CONFIDENCE,
                    )
                    frame_counts = build_counts_from_detections(final_frame_dets)
                    for d in final_frame_dets:
                        d2 = dict(d)
                        d2['frame'] = int(fidx)
                        frames_info.append(d2)

                    if first_annotated is None:
                        first_annotated = overlay_detections(frame_img, final_frame_dets, conf_thresh=MIN_COUNT_CONFIDENCE)
                        preview_counts = frame_counts
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
        predict_kwargs = dict(source=frames_batch, conf=max(float(conf), MIN_COUNT_CONFIDENCE), imgsz=imgsz)
        if use_fp16 and (torch is not None and torch.cuda.is_available()):
            predict_kwargs['half'] = True
        try:
            predict_kwargs['iou'] = float(st.session_state.get('nms_iou', 0.5))
            predict_kwargs['max_det'] = int(st.session_state.get('max_det', 100))
        except Exception:
            pass
        results_batch = model.predict(**predict_kwargs)
        for i, results in enumerate(results_batch):
            names = results[0].names if hasattr(results[0], 'names') else {}
            fidx = frames_batch_idx[i]
            raw_frame_dets = []
            for r in results:
                boxes = getattr(r, 'boxes', None)
                if boxes is None:
                    continue
                for box in boxes:
                    confv = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    if confv < MIN_COUNT_CONFIDENCE:
                        continue
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else None
                    name = normalize_class_name(names.get(cls, str(cls)))
                    xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else None
                    raw_frame_dets.append({
                        "frame": int(fidx),
                        "class": name,
                        "conf": float(confv),
                        "xyxy": xyxy.tolist() if xyxy is not None else None,
                    })

            frame_img = frames_batch[i]
            fh, fw = frame_img.shape[:2] if isinstance(frame_img, np.ndarray) else (0, 0)
            final_frame_dets = finalize_frame_detections_for_count(
                raw_frame_dets,
                img_w=int(fw),
                img_h=int(fh),
                min_conf=MIN_COUNT_CONFIDENCE,
            )
            frame_counts = build_counts_from_detections(final_frame_dets)
            for d in final_frame_dets:
                d2 = dict(d)
                d2['frame'] = int(fidx)
                frames_info.append(d2)

            if first_annotated is None:
                first_annotated = overlay_detections(frame_img, final_frame_dets, conf_thresh=MIN_COUNT_CONFIDENCE)
                preview_counts = frame_counts

    if first_annotated is None:
        first_annotated = np.zeros((480, 640, 3), dtype=np.uint8)
    return first_annotated, preview_counts, frames_info


def process_uploaded(
    model,
    uploaded_bytes: bytes,
    is_image: bool,
    conf: float,
    imgsz: int = 640,
    use_fp16: bool = True,
    max_frames: int = 30,
    sample_rate: int = 5,
    batch_size: int = 4,
    nms_iou: float = None,
    max_det: int = None,
    prefer_recall_custom: bool = False,
    mode: str = None,
):
    def _fast_finalize_custom_image(img_np, dets_in, tuned_conf_local, stage_stats_local):
        """Fast/stable custom-only postprocess.

        Goal: avoid over-complex cascades that create random boxes, while keeping
        one-object-one-box behavior for Custom4 classes.
        """
        ih, iw = img_np.shape[:2] if isinstance(img_np, np.ndarray) else (0, 0)
        dets = canonicalize_final_detections(dets_in)
        dets = ensure_source_model(dets, "custom")
        dets = filter_detections_by_mode(dets, "custom")

        # Class floors tuned for recall + fewer random low-confidence ghosts.
        class_floor = {
            'flower': max(0.24, float(tuned_conf_local) * 0.82),
            'fruit': max(0.20, float(tuned_conf_local) * 0.72),
            'tree': max(0.18, float(tuned_conf_local) * 0.65),
            'dumbbell': max(0.14, float(tuned_conf_local) * 0.60),
        }
        raw_flower_n = int(sum(1 for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower'))
        raw_flower_candidates = [
            dict(d) for d in dets
            if (normalize_class_name(d.get('class', '')) or '') == 'flower'
        ]
        raw_fruit_candidates = [
            dict(d) for d in dets
            if (normalize_class_name(d.get('class', '')) or '') in ({'fruit'} | set(SPECIFIC_FRUIT_CLASSES))
        ]
        raw_fruit_n = len(raw_fruit_candidates)
        raw_tree_candidates = [
            dict(d) for d in dets
            if (normalize_class_name(d.get('class', '')) or '') == 'tree'
        ]
        raw_tree_n = len(raw_tree_candidates)
        pre = []
        for d in dets:
            cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
            confv = float(d.get('conf', 0.0))
            if confv < float(class_floor.get(cls_name, max(0.25, float(tuned_conf_local)))):
                continue
            area_ratio, _, _ = _box_metrics(d.get('xyxy'), iw, ih)
            if area_ratio <= 0:
                continue
            if cls_name == 'flower' and area_ratio >= 0.55 and confv < 0.88:
                continue
            if cls_name == 'flower' and area_ratio >= 0.28 and confv < 0.44:
                continue
            if cls_name == 'tree' and area_ratio >= 0.70 and confv < 0.72:
                continue
            if cls_name == 'tree' and area_ratio >= 0.45 and confv < 0.50:
                continue
            if cls_name == 'tree' and area_ratio >= 0.25 and confv < 0.34:
                continue
            if cls_name == 'fruit' and area_ratio >= 0.18 and confv < 0.58:
                continue
            if cls_name == 'dumbbell':
                _, aspect, _ = _box_metrics(d.get('xyxy'), iw, ih)
                if aspect < 1.05 and confv < 0.60:
                    continue
                # Reject person-like dumbbell hallucinations (skin-dominant ROI).
                try:
                    x1, y1, x2, y2 = map(int, d.get('xyxy')[:4])
                    x1 = max(0, min(iw - 1, x1))
                    x2 = max(0, min(iw, x2))
                    y1 = max(0, min(ih - 1, y1))
                    y2 = max(0, min(ih, y2))
                    if x2 > x1 and y2 > y1:
                        roi = img_np[y1:y2, x1:x2]
                        if roi is not None and roi.size > 0:
                            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                            ycc = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
                            hch = hsv[:, :, 0]
                            sch = hsv[:, :, 1]
                            ych = ycc[:, :, 0]
                            cr = ycc[:, :, 1]
                            cb = ycc[:, :, 2]
                            skin_mask = (
                                (ych >= 40) & (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
                                & (hch <= 25) & (sch >= 25)
                            )
                            skin_ratio = float(np.count_nonzero(skin_mask)) / float(max(1, roi.shape[0] * roi.shape[1]))
                            if skin_ratio >= 0.22 and confv < 0.85:
                                continue
                except Exception:
                    pass
            pre.append(d)

        pre = suppress_scene_level_boxes(pre, img_w=iw, img_h=ih)
        pre = refine_flower_boxes_with_visual_evidence(img_np, pre, img_w=iw, img_h=ih)

        # If flowers exist, suppress weak dumbbell ghosts in flower scenes.
        flower_n_pre = int(sum(1 for d in pre if (normalize_class_name(d.get('class', '')) or '') == 'flower'))
        if flower_n_pre >= 1:
            pre2 = []
            for d in pre:
                cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
                confv = float(d.get('conf', 0.0))
                if cls_name == 'dumbbell' and flower_n_pre >= 2 and confv < 0.58:
                    continue
                pre2.append(d)
            pre = pre2

        if raw_flower_n >= 2:
            pre_flower_n = int(
                sum(
                    1 for d in pre
                    if (normalize_class_name(d.get('class', '')) or '') == 'flower'
                )
            )
            if pre_flower_n <= max(1, raw_flower_n // 2):
                flower_rescue = []
                for d in raw_flower_candidates:
                    confv = float(d.get('conf', 0.0))
                    area_ratio, _, near_full = _box_metrics(d.get('xyxy'), iw, ih)
                    if area_ratio <= 0:
                        continue
                    if near_full and confv < 0.86:
                        continue
                    if area_ratio >= 0.40 and confv < 0.52:
                        continue
                    if area_ratio < 0.0010 and confv < 0.32:
                        continue
                    if confv < max(0.22, float(tuned_conf_local) * 0.58):
                        continue
                    flower_rescue.append(dict(d))
                if flower_rescue:
                    flower_rescue = suppress_face_like_flower_boxes(img_np, flower_rescue, img_w=iw, img_h=ih)
                    flower_rescue = suppress_flower_boxes_on_faces(img_np, flower_rescue, img_w=iw, img_h=ih)
                    flower_rescue = suppress_flower_on_fruit_confusions(flower_rescue, img_w=iw, img_h=ih, image_np=img_np)
                    flower_rescue = collapse_sparse_flower_duplicates(flower_rescue, img_w=iw, img_h=ih)
                    if len(flower_rescue) > pre_flower_n:
                        non_flower_pre = [
                            d for d in pre
                            if (normalize_class_name(d.get('class', '')) or '') != 'flower'
                        ]
                        pre = dedup_detections_by_class_nms_classwise(non_flower_pre + flower_rescue, default_iou=0.74)

        pre_fruit_n = int(
            sum(
                1 for d in pre
                if (normalize_class_name(d.get('class', '')) or '') in ({'fruit'} | set(SPECIFIC_FRUIT_CLASSES))
            )
        )
        if raw_fruit_n >= 3 and pre_fruit_n <= 1:
            fruit_rescue = []
            for d in raw_fruit_candidates:
                confv = float(d.get('conf', 0.0))
                area_ratio, _, near_full = _box_metrics(d.get('xyxy'), iw, ih)
                if area_ratio <= 0:
                    continue
                if near_full and confv < 0.88:
                    continue
                if area_ratio >= 0.24 and confv < 0.54:
                    continue
                if confv < max(0.18, float(tuned_conf_local) * 0.52):
                    continue
                fruit_rescue.append(dict(d))
            if fruit_rescue:
                pre = dedup_detections_by_class_nms_classwise(pre + fruit_rescue, default_iou=0.72)

        pre_tree_n = int(
            sum(
                1 for d in pre
                if (normalize_class_name(d.get('class', '')) or '') == 'tree'
            )
        )
        if raw_tree_n >= 2 and pre_tree_n <= 1:
            tree_rescue = []
            for d in raw_tree_candidates:
                confv = float(d.get('conf', 0.0))
                area_ratio, _, near_full = _box_metrics(d.get('xyxy'), iw, ih)
                if area_ratio <= 0:
                    continue
                if near_full and confv < 0.82:
                    continue
                if area_ratio >= 0.45 and confv < 0.34:
                    continue
                if area_ratio < 0.003 and confv < 0.24:
                    continue
                if confv < max(0.16, float(tuned_conf_local) * 0.48):
                    continue
                tree_rescue.append(dict(d))
            if tree_rescue:
                pre = dedup_detections_by_class_nms_classwise(pre + tree_rescue, default_iou=0.70)

        should_try_flower_recovery = (
            (raw_flower_n >= 1 and flower_n_pre == 0)
            or (raw_flower_n >= 3 and flower_n_pre <= max(1, raw_flower_n // 2))
        )
        if should_try_flower_recovery:
            try:
                rescued = recover_flower_instances(
                    model=model,
                    image_np=img_np,
                    base_filtered=pre,
                    tuned_conf=float(tuned_conf_local),
                    imgsz=int(imgsz),
                    use_fp16=bool(use_fp16),
                )
                rescued = refine_flower_boxes_with_visual_evidence(img_np, rescued, img_w=iw, img_h=ih)
                rescued = filter_detections_by_mode(rescued, "custom")
                rescued_flower_n = int(
                    sum(1 for d in rescued if (normalize_class_name(d.get('class', '')) or '') == 'flower')
                )
                # Only accept rescue when it materially improves flower recall.
                if rescued and rescued_flower_n >= max(2, flower_n_pre + 1):
                    pre = rescued
                    flower_n_pre = rescued_flower_n
            except Exception:
                pass

        # Strict finalization for counting/rendering consistency.
        finalized = finalize_frame_detections_for_count(
            pre,
            img_w=iw,
            img_h=ih,
            min_conf=float(max(0.18, float(tuned_conf_local) * 0.70)),
        )
        finalized = filter_detections_by_mode(finalized, "custom")

        # Cross-class sanity in custom-only mode:
        # reduce obvious confusion spill (flower<->dumbbell, fruit->flower/tree).
        ctx_counts = build_counts_from_detections(finalized)
        flower_n = int(ctx_counts.get('flower', 0))
        fruit_n = int(ctx_counts.get('fruit', 0))
        tree_n = int(ctx_counts.get('tree', 0))
        dumbbell_n = int(ctx_counts.get('dumbbell', 0))

        # Precision guard: a single weak/large dumbbell in a non-custom scene is likely false.
        if dumbbell_n == 1 and flower_n == 0 and fruit_n == 0 and tree_n == 0:
            db_only = [d for d in finalized if (normalize_class_name(d.get('class', '')) or '') == 'dumbbell']
            if db_only:
                db = db_only[0]
                db_conf = float(db.get('conf', 0.0))
                db_area, _, _ = _box_metrics(db.get('xyxy'), iw, ih)
                if db_conf < 0.32 and db_area >= 0.05:
                    finalized = []
                    dumbbell_n = 0

        if flower_n >= 2 or fruit_n >= 1 or dumbbell_n >= 1:
            cleaned = []
            fruit_refs = [
                d for d in finalized
                if (normalize_class_name(d.get('class', '')) or '') == 'fruit'
            ]
            for d in finalized:
                cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
                confv = float(d.get('conf', 0.0))
                area_ratio, aspect, _ = _box_metrics(d.get('xyxy'), iw, ih)
                if cls_name == 'dumbbell' and flower_n >= 2:
                    if confv < 0.58 and (area_ratio < 0.0015 or aspect < 1.08):
                        continue
                if cls_name == 'flower' and fruit_n >= 1:
                    overlap_fruit = any(_box_iou(d.get('xyxy'), f.get('xyxy')) >= 0.05 for f in fruit_refs)
                    # In fruit scenes, flower is usually noise unless very certain and non-overlapping.
                    if overlap_fruit and confv < 0.95:
                        continue
                    if fruit_n >= 3 and confv < 0.90:
                        continue
                if cls_name == 'flower' and dumbbell_n >= 1 and confv < 0.88:
                    continue
                if cls_name == 'flower' and tree_n >= 1 and fruit_n == 0 and dumbbell_n == 0:
                    # In tree-focused scenes, keep only strong flower evidence.
                    if confv < 0.78 or area_ratio < 0.005:
                        continue
                if cls_name == 'tree' and fruit_n >= 3 and confv < 0.70:
                    continue
                cleaned.append(d)
            finalized = cleaned

        if raw_tree_n >= 2:
            tree_final_n = int(
                sum(
                    1 for d in finalized
                    if (normalize_class_name(d.get('class', '')) or '') == 'tree'
                )
            )
            if tree_final_n <= 1:
                tree_restore = []
                for d in raw_tree_candidates:
                    confv = float(d.get('conf', 0.0))
                    area_ratio, _, near_full = _box_metrics(d.get('xyxy'), iw, ih)
                    if area_ratio <= 0:
                        continue
                    if near_full and confv < 0.84:
                        continue
                    if area_ratio >= 0.45 and confv < 0.36:
                        continue
                    if area_ratio < 0.003 and confv < 0.24:
                        continue
                    if confv < max(0.16, float(tuned_conf_local) * 0.48):
                        continue
                    tree_restore.append(dict(d))
                if tree_restore:
                    non_tree = [
                        d for d in finalized
                        if (normalize_class_name(d.get('class', '')) or '') != 'tree'
                    ]
                    finalized = dedup_detections_by_class_nms_classwise(non_tree + tree_restore, default_iou=0.72)

        if raw_fruit_n >= 3:
            fruit_final_n = int(
                sum(
                    1 for d in finalized
                    if (normalize_class_name(d.get('class', '')) or '') in ({'fruit'} | set(SPECIFIC_FRUIT_CLASSES))
                )
            )
            if fruit_final_n <= 1:
                fruit_restore = []
                for d in raw_fruit_candidates:
                    confv = float(d.get('conf', 0.0))
                    area_ratio, _, near_full = _box_metrics(d.get('xyxy'), iw, ih)
                    if area_ratio <= 0:
                        continue
                    if near_full and confv < 0.90:
                        continue
                    if area_ratio >= 0.24 and confv < 0.56:
                        continue
                    if confv < max(0.18, float(tuned_conf_local) * 0.52):
                        continue
                    fruit_restore.append(dict(d))
                if fruit_restore:
                    non_fruit = [
                        d for d in finalized
                        if (normalize_class_name(d.get('class', '')) or '') not in ({'fruit'} | set(SPECIFIC_FRUIT_CLASSES))
                    ]
                    finalized = dedup_detections_by_class_nms_classwise(non_fruit + fruit_restore, default_iou=0.74)

        finalized = dedup_detections_by_class_nms_classwise(finalized, default_iou=0.60)
        finalized = suppress_scene_level_boxes(finalized, img_w=iw, img_h=ih)
        finalized = enforce_one_object_one_box(finalized, img_w=iw, img_h=ih)
        finalized = trim_sparse_custom_outliers(finalized, img_w=iw, img_h=ih)
        finalized = suppress_face_like_flower_boxes(img_np, finalized, img_w=iw, img_h=ih)
        finalized = suppress_flower_boxes_on_faces(img_np, finalized, img_w=iw, img_h=ih)
        finalized = suppress_flower_on_fruit_confusions(finalized, img_w=iw, img_h=ih, image_np=img_np)
        finalized = suppress_dumbbell_on_human_faces(img_np, finalized, img_w=iw, img_h=ih)
        finalized = recenter_flower_boxes_by_visual_evidence(img_np, finalized)
        finalized = collapse_sparse_flower_duplicates(finalized, img_w=iw, img_h=ih)
        finalized = dedup_detections_by_class_nms_classwise(finalized, default_iou=0.58)

        # Fallback: avoid ending with empty output when pre-stage had a reasonable candidate.
        if not finalized and pre:
            pre_sorted = sorted(pre, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
            recovered_one = None
            for d in pre_sorted:
                cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
                confv = float(d.get('conf', 0.0))
                area_ratio, _, near_full = _box_metrics(d.get('xyxy'), iw, ih)
                if area_ratio <= 0:
                    continue
                if near_full and confv < 0.92:
                    continue
                if cls_name == 'flower' and confv < 0.36:
                    continue
                if cls_name == 'fruit' and confv < 0.30:
                    continue
                if cls_name == 'tree' and confv < 0.28:
                    continue
                if cls_name == 'dumbbell' and confv < 0.32:
                    continue
                recovered_one = dict(d)
                break
            if recovered_one is not None:
                finalized = [recovered_one]

        finalized_flower_n = int(
            sum(1 for d in finalized if (normalize_class_name(d.get('class', '')) or '') == 'flower')
        )
        should_last_chance_flower_rescue = (
            finalized_flower_n == 0
            and (
                raw_flower_n >= 1
                or (not finalized and raw_fruit_n == 0 and raw_tree_n == 0)
            )
        )
        if should_last_chance_flower_rescue:
            try:
                rescue_seed = pre if pre else dets
                rescued = recover_flower_instances(
                    model=model,
                    image_np=img_np,
                    base_filtered=rescue_seed,
                    tuned_conf=float(tuned_conf_local),
                    imgsz=int(max(int(imgsz), 640)),
                    use_fp16=bool(use_fp16),
                )
                rescued = filter_detections_by_mode(rescued, "custom")
                rescued = suppress_scene_level_boxes(rescued, img_w=iw, img_h=ih)
                rescued = enforce_one_object_one_box(rescued, img_w=iw, img_h=ih)
                rescued = trim_sparse_custom_outliers(rescued, img_w=iw, img_h=ih)
                rescued = suppress_face_like_flower_boxes(img_np, rescued, img_w=iw, img_h=ih)
                rescued = suppress_flower_boxes_on_faces(img_np, rescued, img_w=iw, img_h=ih)
                rescued = suppress_flower_on_fruit_confusions(rescued, img_w=iw, img_h=ih, image_np=img_np)
                rescued = recenter_flower_boxes_by_visual_evidence(img_np, rescued)
                rescued = collapse_sparse_flower_duplicates(rescued, img_w=iw, img_h=ih)
                rescued = dedup_detections_by_class_nms_classwise(rescued, default_iou=0.58)

                rescued_flower = [
                    d for d in rescued
                    if (normalize_class_name(d.get('class', '')) or '') == 'flower'
                ]
                rescued_flower_n = len(rescued_flower)
                rescued_flower_max_conf = max((float(d.get('conf', 0.0)) for d in rescued_flower), default=0.0)
                rescued_flower_avg_conf = (
                    sum(float(d.get('conf', 0.0)) for d in rescued_flower) / float(max(1, rescued_flower_n))
                )
                rescue_limit = 4 if raw_flower_n == 0 else max(6, raw_flower_n + 2)
                rescue_conf_floor = 0.30 if raw_flower_n == 0 else 0.22
                accept_flower_rescue = (
                    rescued_flower_n >= max(1, finalized_flower_n + 1)
                    and rescued_flower_n <= rescue_limit
                    and rescued_flower_max_conf >= rescue_conf_floor
                    and rescued_flower_avg_conf >= max(0.20, rescue_conf_floor - 0.06)
                )
                if accept_flower_rescue:
                    finalized = rescued
            except Exception:
                pass

        # Keep stage stats for debug panel.
        stage_stats_local['fast_custom_pre'] = summarize_detection_stage(pre)
        stage_stats_local['final'] = summarize_detection_stage(finalized)
        ann_local = overlay_detections(img_np, finalized, conf_thresh=float(max(0.20, tuned_conf_local)))
        counts_local = build_counts_from_detections(finalized)
        return ann_local, counts_local, finalized

    if is_image:
        pil = Image.open(io.BytesIO(uploaded_bytes))
        img = np.array(pil.convert('RGB'))
        # expose current image for scene-level checks in downstream filters
        try:
            globals()['CURRENT_IMAGE_FOR_SCENE_CHECK'] = img
        except Exception:
            pass
        stage_stats = {}
        base_nms_iou = float(nms_iou) if nms_iou is not None else float(st.session_state.get('nms_iou', 0.5))
        base_max_det = int(max_det) if max_det is not None else int(st.session_state.get('max_det', 100))
        tuned_conf, tuned_nms_iou, tuned_max_det = auto_tune_image_inference_params(
            img,
            base_conf=float(conf),
            base_nms_iou=base_nms_iou,
            base_max_det=base_max_det,
        )
        # Custom-only path: allow slightly lower confidence and more permissive NMS
        # to improve recall for tree/fruit/flower counts.
        if prefer_recall_custom:
            tuned_conf = min(float(tuned_conf), max(0.16, float(conf) * 0.60))
            tuned_nms_iou = min(0.85, max(0.50, float(tuned_nms_iou) + 0.05))
            tuned_max_det = max(int(tuned_max_det), 160)

        ann, counts, detections = run_detection_on_image(
            model,
            img,
            tuned_conf,
            imgsz=imgsz,
            use_fp16=use_fp16,
            nms_iou=tuned_nms_iou,
            max_det=tuned_max_det,
        )
        detections = canonicalize_final_detections(detections)
        
        # FILTER#1: Apply mode-based source filtering after raw model prediction
        if mode is None:
            mode = "custom" if prefer_recall_custom else "coco"
        detections = ensure_source_model(detections, mode if mode != "hybrid" else "coco")
        detections = filter_detections_by_mode(detections, mode)
        
        stage_stats['raw_model'] = summarize_detection_stage(detections)

        # Fast path for Custom4-only mode:
        # return after the first model pass and lightweight finalization to keep UI responsive.
        if mode == "custom":
            filtered_custom = filter_detections_by_mode(detections, "custom")
            ann, counts, filtered_custom = _fast_finalize_custom_image(
                img,
                filtered_custom,
                tuned_conf,
                stage_stats,
            )
            try:
                if 'CURRENT_IMAGE_FOR_SCENE_CHECK' in globals():
                    del globals()['CURRENT_IMAGE_FOR_SCENE_CHECK']
            except Exception:
                pass
            return ann, counts, {
                "detections": filtered_custom,
                "auto_params": {
                    "conf": float(tuned_conf),
                    "nms_iou": float(tuned_nms_iou),
                    "max_det": int(tuned_max_det),
                },
                "stage_stats": stage_stats,
            }, img

        # If custom dumbbell is supported, allow a slightly lower tuned conf to improve recall.
        try:
            if 'dumbbell' in get_custom_class_names():
                tuned_conf = min(float(tuned_conf), 0.22)
        except Exception:
            pass

        # Adaptive re-run for unstable first pass:
        # - too many low-confidence boxes -> stricter pass
        # - too few boxes -> permissive pass
        raw_n = len(detections)
        raw_avg_conf = (sum(float(d.get('conf', 0.0)) for d in detections) / raw_n) if raw_n > 0 else 0.0
        if raw_n >= 20 and raw_avg_conf < 0.62:
            strict_conf = min(0.82, tuned_conf + 0.08)
            strict_iou = max(0.45, tuned_nms_iou - 0.03)
            strict_max_det = max(50, int(tuned_max_det * 0.85))
            _, _, strict_dets = run_detection_on_image(
                model,
                img,
                strict_conf,
                imgsz=imgsz,
                use_fp16=use_fp16,
                nms_iou=strict_iou,
                max_det=strict_max_det,
            )
            strict_dets = canonicalize_final_detections(strict_dets)
            if len(strict_dets) <= max(1, int(raw_n * 0.92)):
                strict_dets = ensure_source_model(strict_dets, mode if mode != "hybrid" else "coco")
                strict_dets = filter_detections_by_mode(strict_dets, mode)
                detections = strict_dets
                tuned_conf = strict_conf
                tuned_nms_iou = strict_iou
                tuned_max_det = strict_max_det
        elif raw_n <= 1:
            permissive_conf = max(0.15, tuned_conf - 0.10)
            permissive_iou = min(0.50, tuned_nms_iou + 0.02)
            permissive_max_det = min(200, int(tuned_max_det * 1.10))
            _, _, permissive_dets = run_detection_on_image(
                model,
                img,
                permissive_conf,
                imgsz=imgsz,
                use_fp16=use_fp16,
                nms_iou=permissive_iou,
                max_det=permissive_max_det,
            )
            permissive_dets = canonicalize_final_detections(permissive_dets)
            if len(permissive_dets) > raw_n:
                permissive_dets = ensure_source_model(permissive_dets, mode if mode != "hybrid" else "coco")
                permissive_dets = filter_detections_by_mode(permissive_dets, mode)
                detections = permissive_dets
                tuned_conf = permissive_conf
                tuned_nms_iou = permissive_iou
                tuned_max_det = permissive_max_det
        stage_stats['after_adaptive_rerun'] = summarize_detection_stage(detections)
        # Dumbbell rescue: when no dumbbell is found, retry a permissive pass.
        try:
            has_dumbbell = any(
                (normalize_class_name(d.get('class', '')) or '') == 'dumbbell'
                for d in detections
            )
            if (not has_dumbbell) and ('dumbbell' in get_custom_class_names()):
                rescue_conf = max(0.12, float(tuned_conf) * 0.50)
                rescue_iou = min(0.55, float(tuned_nms_iou) + 0.04)
                rescue_max_det = max(150, int(tuned_max_det))
                rescue_imgsz = int(max(int(imgsz), 768))
                _, _, rescue_dets = run_detection_on_image(
                    model,
                    img,
                    rescue_conf,
                    imgsz=rescue_imgsz,
                    use_fp16=use_fp16,
                    nms_iou=rescue_iou,
                    max_det=rescue_max_det,
                )
                rescue_dets = [
                    d for d in canonicalize_final_detections(rescue_dets)
                    if (normalize_class_name(d.get('class', '')) or '') == 'dumbbell'
                    and float(d.get('conf', 0.0)) >= 0.18
                ]
                if rescue_dets:
                    rescue_dets = ensure_source_model(rescue_dets, mode if mode != "hybrid" else "coco")
                    rescue_dets = filter_detections_by_mode(rescue_dets, mode)
                    if rescue_dets:
                        detections = dedup_detections_by_class_nms_classwise(
                            detections + rescue_dets,
                            default_iou=0.60,
                        )
                    stage_stats['after_dumbbell_rescue'] = summarize_detection_stage(detections)

            has_tree = any(
                (normalize_class_name(d.get('class', '')) or '') == 'tree'
                for d in detections
            )
            has_fruit_any = any(
                (normalize_class_name(d.get('class', '')) or '') == 'fruit'
                or (normalize_class_name(d.get('class', '')) or '') in SPECIFIC_FRUIT_CLASSES
                for d in detections
            )
            if (not has_tree) and (not has_dumbbell) and (not has_fruit_any) and ('tree' in get_custom_class_names()):
                rescue_conf = max(0.12, float(tuned_conf) * 0.50)
                rescue_iou = min(0.55, float(tuned_nms_iou) + 0.04)
                rescue_max_det = max(150, int(tuned_max_det))
                rescue_imgsz = int(max(int(imgsz), 768))
                _, _, rescue_dets = run_detection_on_image(
                    model,
                    img,
                    rescue_conf,
                    imgsz=rescue_imgsz,
                    use_fp16=use_fp16,
                    nms_iou=rescue_iou,
                    max_det=rescue_max_det,
                )
                rescue_dets = [
                    d for d in canonicalize_final_detections(rescue_dets)
                    if (normalize_class_name(d.get('class', '')) or '') == 'tree'
                    and float(d.get('conf', 0.0)) >= 0.18
                ]
                if rescue_dets:
                    rescue_dets = ensure_source_model(rescue_dets, mode if mode != "hybrid" else "coco")
                    rescue_dets = filter_detections_by_mode(rescue_dets, mode)
                    if rescue_dets:
                        detections = dedup_detections_by_class_nms_classwise(
                            detections + rescue_dets,
                            default_iou=0.60,
                        )
                    stage_stats['after_tree_rescue'] = summarize_detection_stage(detections)
        except Exception:
            pass

        # Last-resort dumbbell-only pass for very low-confidence cases (e.g., product ads).
        try:
            if not detections and ('dumbbell' in get_custom_class_names()):
                final_conf = 0.08
                final_iou = min(0.55, float(tuned_nms_iou) + 0.05)
                final_max_det = max(200, int(tuned_max_det))
                final_imgsz = int(max(int(imgsz), 960))
                _, _, final_dets = run_detection_on_image(
                    model,
                    img,
                    final_conf,
                    imgsz=final_imgsz,
                    use_fp16=use_fp16,
                    nms_iou=final_iou,
                    max_det=final_max_det,
                )
                final_dets = [
                    d for d in canonicalize_final_detections(final_dets)
                    if (normalize_class_name(d.get('class', '')) or '') == 'dumbbell'
                    and float(d.get('conf', 0.0)) >= 0.12
                ]
                if final_dets:
                    final_dets = ensure_source_model(final_dets, mode if mode != "hybrid" else "coco")
                    final_dets = filter_detections_by_mode(final_dets, mode)
                    if final_dets:
                        detections = final_dets
                    stage_stats['after_dumbbell_last_resort'] = summarize_detection_stage(detections)
        except Exception:
            pass
        raw_for_rescue = canonicalize_final_detections(detections)

        # Dense-flower fallback: if full-image inference collapses to one large flower box,
        # run tiled inference and merge to improve object counting.
        img_h, img_w = img.shape[:2]
        flower_dets = [d for d in detections if str(d.get('class', '')).lower().strip() == 'flower']
        has_large_flower_box = False
        small_flower_n = 0
        medium_flower_n = 0
        max_flower_area = 0.0
        coverage_gap = False
        for d in flower_dets:
            area_ratio, _, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)
            max_flower_area = max(max_flower_area, float(area_ratio))
            if area_ratio > 0 and area_ratio <= 0.08 and float(d.get('conf', 0.0)) >= max(0.20, float(tuned_conf) * 0.75):
                small_flower_n += 1
            if area_ratio >= 0.008 and area_ratio <= 0.18 and float(d.get('conf', 0.0)) >= max(0.18, float(tuned_conf) * 0.65):
                medium_flower_n += 1
            if near_full or area_ratio >= 0.38:
                has_large_flower_box = True
                break

        # Coverage-gap signal: flower boxes only cover a partial region of the image.
        union_ratio = 1.0
        if len(flower_dets) >= 6 and img_w > 0 and img_h > 0:
            try:
                min_x = float('inf')
                min_y = float('inf')
                max_x = float('-inf')
                max_y = float('-inf')
                for d in flower_dets:
                    x1, y1, x2, y2 = map(float, d.get('xyxy')[:4])
                    min_x = min(min_x, x1)
                    min_y = min(min_y, y1)
                    max_x = max(max_x, x2)
                    max_y = max(max_y, y2)
                union_area = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
                union_ratio = union_area / float(max(1.0, img_w * img_h))
                if union_ratio <= 0.60 and not has_large_flower_box:
                    coverage_gap = True
            except Exception:
                coverage_gap = False

        total_dets = max(1, len(detections))
        flower_ratio = len(flower_dets) / float(total_dets)
        auto_dense_scene = (
            (len(flower_dets) >= 5 and flower_ratio >= 0.60 and (small_flower_n >= 3 or medium_flower_n >= 4))
            or (len(flower_dets) <= 2 and has_large_flower_box)
            or coverage_gap
        )
        # Under-count guard for clustered flowers: trigger tiled pass earlier.
        suspect_dense_scene = (
            len(flower_dets) >= 2
            and len(flower_dets) <= 5
            and flower_ratio >= 0.55
            and small_flower_n >= 2
        )

        # Extra under-count signal: several medium flower boxes but total flower count still low.
        suspect_dense_scene = suspect_dense_scene or (
            len(flower_dets) <= 4
            and flower_ratio >= 0.55
            and medium_flower_n >= 2
        )
        # Bouquet / close-up flowers: few large flower boxes likely indicate under-count.
        suspect_dense_scene = suspect_dense_scene or (
            len(flower_dets) >= 2
            and len(flower_dets) <= 6
            and flower_ratio >= 0.45
            and max_flower_area >= 0.12
        )
        suspect_dense_scene = suspect_dense_scene or coverage_gap

        # Important: allow suspect scenes to trigger dense fallback automatically.
        enable_dense = (
            bool(st.session_state.get("enable_dense_flower_fallback", ENABLE_DENSE_FLOWER_FALLBACK))
            or auto_dense_scene
            or suspect_dense_scene
        )
        explicit_flower_recovery = bool(st.session_state.get("force_flower_recovery", False))

        # If coverage_gap is detected, always force a tiled flower pass
        # (coverage_gap indicates under-coverage of flower boxes in the full-image pass).
        force_tiled_flower_pass = bool(coverage_gap)
        if force_tiled_flower_pass or (enable_dense and (auto_dense_scene or suspect_dense_scene or (len(flower_dets) <= 1 and has_large_flower_box))):
            tiled_flower = run_tiled_flower_detections(
                model,
                img,
                base_conf=float(tuned_conf),
                imgsz=imgsz,
                use_fp16=use_fp16,
            )
            if len(tiled_flower) > len(flower_dets) or coverage_gap:
                merged = detections + tiled_flower
                dedup_iou = 0.92 if coverage_gap else 0.88
                detections = dedup_detections_by_class_nms_classwise(merged, default_iou=dedup_iou)
        stage_stats['after_dense_fallback'] = summarize_detection_stage(detections)

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
            base_conf=float(tuned_conf),
            strict_coco=False,
            scene_rule_debug=stage_stats,
        )
        stage_stats['after_verify_reduce'] = summarize_detection_stage(filtered)

        # Root-fix: never let strict post-filtering erase all valid detections.
        if not filtered and pre_verified:
            filtered = pre_verified

        # Last-resort fallback from raw detections with permissive confidence.
        if not filtered and detections:
            fallback = []
            min_conf = max(0.08, float(tuned_conf) * 0.45)
            for d in canonicalize_final_detections(detections):
                confv = float(d.get('conf', 0.0))
                if confv < min_conf:
                    continue
                area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
                if area_ratio <= 0:
                    continue
                fallback.append(d)
            filtered = dedup_detections_by_class_nms_classwise(fallback, default_iou=0.65)
        stage_stats['after_fallback'] = summarize_detection_stage(filtered)

        # Remove whole-scene scan boxes when enough local objects already exist.
        filtered = suppress_scene_level_boxes(filtered, img_w=img_w, img_h=img_h)

        # Deterministic flower counting recovery: dedicated permissive pass + center clustering.
        if enable_dense or explicit_flower_recovery:
            filtered = recover_flower_instances(
                model=model,
                image_np=img,
                base_filtered=filtered,
                tuned_conf=float(tuned_conf),
                imgsz=int(imgsz),
                use_fp16=use_fp16,
            )
            stage_stats['flower_recovery_enabled'] = True
        else:
            stage_stats['flower_recovery_enabled'] = False
        filtered = suppress_scene_level_boxes(filtered, img_w=img_w, img_h=img_h)
        filtered = enforce_one_object_one_box(filtered, img_w=img_w, img_h=img_h)
        filtered = _apply_scene_context_rules(
            filtered,
            img_w=img_w,
            img_h=img_h,
            base_conf=float(tuned_conf),
        )
        stage_stats['after_recovery_cluster'] = summarize_detection_stage(filtered)

        # Dense-flower rescue: if still under-counted in clustered flower scenes,
        # run an additional permissive tiled pass and merge only flower boxes.
        flower_after = [d for d in filtered if (normalize_class_name(d.get('class', '')) or '') == 'flower']
        if (auto_dense_scene or suspect_dense_scene) and len(flower_after) <= 5:
            rescue_conf = max(0.12, float(tuned_conf) * 0.60)
            rescue_imgsz = int(max(int(imgsz), 768))
            rescue_flower = run_tiled_flower_detections(
                model,
                img,
                base_conf=float(rescue_conf),
                imgsz=rescue_imgsz,
                use_fp16=use_fp16,
            )
            rescue_clean = []
            for d in rescue_flower:
                confv = float(d.get('conf', 0.0))
                if confv < max(0.18, float(tuned_conf) * 0.50):
                    continue
                area_ratio, _, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)
                if area_ratio <= 0:
                    continue
                if area_ratio < 0.00016 and confv < 0.62:
                    continue
                if near_full and confv < 0.90:
                    continue
                rescue_clean.append(d)

            if rescue_clean:
                merged_flower = flower_after + rescue_clean
                merged_flower = dedup_detections_by_class_nms_classwise(merged_flower, default_iou=0.86)
                if len(merged_flower) > len(flower_after):
                    non_flower = [d for d in filtered if (normalize_class_name(d.get('class', '')) or '') != 'flower']
                    filtered = non_flower + merged_flower
                    filtered = suppress_scene_level_boxes(filtered, img_w=img_w, img_h=img_h)
                    filtered = _apply_scene_context_rules(
                        filtered,
                        img_w=img_w,
                        img_h=img_h,
                        base_conf=float(tuned_conf),
                    )

        # Recovery pass: if initial run produced empty outputs, retry once with lower confidence.
        if not filtered:
            retry_conf = max(0.10, min(0.22, float(tuned_conf) * 0.55))
            _, _, retry_dets = run_detection_on_image(
                model,
                img,
                conf=retry_conf,
                imgsz=imgsz,
                use_fp16=use_fp16,
                nms_iou=tuned_nms_iou,
                max_det=tuned_max_det,
            )
            retry_clean = []
            for d in canonicalize_final_detections(retry_dets):
                confv = float(d.get('conf', 0.0))
                if confv < retry_conf:
                    continue
                area_ratio, _, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)
                if area_ratio <= 0:
                    continue
                if area_ratio < 0.00045 and confv < 0.70:
                    continue
                if near_full and confv < 0.80:
                    continue
                retry_clean.append(d)
            filtered = verify_and_reduce_detections(
                retry_clean,
                img_w=img_w,
                img_h=img_h,
                base_conf=float(max(0.16, retry_conf)),
                strict_coco=False,
            )

        filtered = suppress_scene_level_boxes(filtered, img_w=img_w, img_h=img_h)
        filtered = enforce_one_object_one_box(filtered, img_w=img_w, img_h=img_h)
        filtered = normalize_dense_flower_boxes(filtered, img_w=img_w, img_h=img_h)
        filtered = refine_flower_boxes_with_visual_evidence(img, filtered, img_w=img_w, img_h=img_h)
        filtered = trim_sparse_flower_outliers(filtered, img_w=img_w, img_h=img_h)
        filtered = prune_single_flower_tail_noise(filtered, img_w=img_w, img_h=img_h)
        filtered = trim_sparse_custom_outliers(filtered, img_w=img_w, img_h=img_h)
        filtered = collapse_sparse_flower_duplicates(filtered, img_w=img_w, img_h=img_h)
        filtered = suppress_flower_cross_class_confusions(filtered, img_w=img_w, img_h=img_h, base_conf=float(tuned_conf))

        # Flower recovery pass for undercounted scenes (custom-only path).
        try:
            flower_n_pre = int(sum(1 for d in filtered if (normalize_class_name(d.get('class', '')) or '') == 'flower'))
            if (enable_dense or explicit_flower_recovery) and flower_n_pre <= 6:
                recovered = recover_flower_instances(
                    model=model,
                    image_np=img,
                    base_filtered=filtered,
                    tuned_conf=float(tuned_conf),
                    imgsz=int(imgsz),
                    use_fp16=use_fp16,
                )
                recovered = suppress_flower_cross_class_confusions(
                    recovered,
                    img_w=img_w,
                    img_h=img_h,
                    base_conf=float(tuned_conf),
                )
                if len(recovered) > len(filtered):
                    filtered = recovered
                    stage_stats['after_recover'] = summarize_detection_stage(filtered)
        except Exception:
            pass

        # Rescue single-flower misses: keep a strong flower from the raw pass.
        try:
            flower_n_now = int(sum(1 for d in filtered if (normalize_class_name(d.get('class', '')) or '') == 'flower'))
            if flower_n_now == 0 and raw_for_rescue:
                rescue = []
                for d in raw_for_rescue:
                    if (normalize_class_name(d.get('class', '')) or '') != 'flower':
                        continue
                    confv = float(d.get('conf', 0.0))
                    if confv < max(0.55, float(tuned_conf) + 0.10):
                        continue
                    area_ratio, aspect, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)
                    if area_ratio <= 0 or near_full:
                        continue
                    if area_ratio < 0.001 and confv < 0.75:
                        continue
                    if area_ratio > 0.18 and confv < 0.88:
                        continue
                    if aspect > 3.5 and confv < 0.85:
                        continue
                    # Avoid rescuing face/body-like flower boxes.
                    checked = suppress_face_like_flower_boxes(
                        img,
                        [dict(d)],
                        img_w=img_w,
                        img_h=img_h,
                    )
                    if not checked:
                        continue
                    rescue.append(d)
                if rescue:
                    rescue = sorted(rescue, key=lambda x: float(x.get('conf', 0.0)), reverse=True)[:1]
                    filtered = filtered + rescue
                    stage_stats['after_rescue'] = summarize_detection_stage(filtered)
        except Exception:
            pass

        # Final flower-only recount mode for static images when flower undercount is likely.
        flower_n = int(sum(1 for d in filtered if (normalize_class_name(d.get('class', '')) or '') == 'flower'))
        total_n = max(1, len(filtered))
        flower_ratio_final = float(flower_n) / float(total_n)
        if USE_SPECIALIST_MODELS and flower_n <= 12 and flower_n >= 2 and flower_ratio_final >= 0.45:
            flower_model = None
            if os.path.exists(FLOWER_SPECIALIST_MODEL_PATH):
                try:
                    flower_model = load_model(FLOWER_SPECIALIST_MODEL_PATH)
                except Exception:
                    flower_model = None
            if flower_model is None:
                flower_model = model

            recounted = recount_flowers_strict(
                flower_model=flower_model,
                image_np=img,
                base_detections=filtered,
                base_conf=float(tuned_conf),
                imgsz=int(imgsz),
                use_fp16=use_fp16,
            )
            recounted = suppress_flower_cross_class_confusions(
                recounted,
                img_w=img_w,
                img_h=img_h,
                base_conf=float(tuned_conf),
            )

            recounted_flower_n = int(sum(1 for d in recounted if (normalize_class_name(d.get('class', '')) or '') == 'flower'))
            if recounted_flower_n >= flower_n:
                filtered = recounted

        stage_stats['final'] = summarize_detection_stage(filtered)

        # FILTER#3: Apply mode filter before final rendering/counting
        filtered = filter_detections_by_mode(filtered, mode)
        
        counts = {}
        for d in filtered:
            c = str(d.get('class', '')).lower().strip() or 'unknown'
            counts[c] = counts.get(c, 0) + 1
        ann = overlay_detections(img, filtered, conf_thresh=float(max(0.10, tuned_conf * 0.60)))
        try:
            if 'CURRENT_IMAGE_FOR_SCENE_CHECK' in globals():
                del globals()['CURRENT_IMAGE_FOR_SCENE_CHECK']
        except Exception:
            pass

        return ann, counts, {
            "detections": filtered,
            "auto_params": {
                "conf": float(tuned_conf),
                "nms_iou": float(tuned_nms_iou),
                "max_det": int(tuned_max_det),
            },
            "stage_stats": stage_stats,
        }, img
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
        disp = _ascii_safe_label(disp)
        col = _color_for_label(label or str(confv))
        _draw_box_with_label(out, (x1, y1, x2, y2), f"{disp} {confv:.2f}", col)
    return out


def _color_for_label(name: str):
    try:
        if not name:
            name = 'obj'
        cls = normalize_class_name(str(name))
        # High-contrast fixed palette for readability
        palette = {
            'flower': (0, 215, 255),   # gold
            'fruit': (0, 140, 255),    # orange
            'orange': (0, 128, 255),   # deep orange
            'banana': (0, 255, 255),   # yellow
            'apple': (0, 255, 128),    # green
            'tree': (0, 200, 0),       # green
            'dumbbell': (255, 0, 255), # magenta
            'vase': (255, 128, 0),     # teal-ish
        }
        if cls in palette:
            return palette[cls]
        # fallback deterministic color for other classes
        s = sum(ord(c) for c in cls or name)
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


def _ascii_safe_label(text: str) -> str:
    s = str(text or '').strip()
    if not s:
        return 'Unknown'
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('đ', 'd').replace('Đ', 'D')
    return s


def apply_filename_context_guard(detections: list, source_name: str | None) -> list:
    """Apply lightweight filename-based class guardrails for known confusion cases."""
    dets = canonicalize_final_detections(detections)
    if not dets or not source_name:
        return dets

    name_ascii = _ascii_safe_label(os.path.basename(str(source_name))).lower()
    fruit_tokens = (
        'chuoi', 'banana', 'cam', 'orange', 'tao', 'apple',
        'xoai', 'mango', 'nho', 'grape', 'dua', 'fruit', 'trai', 'qua'
    )
    person_tokens = ('nguoi', 'person', 'human', 'selfie', 'people')
    tree_tokens = ('cay', 'tree', 'bonsai')
    has_fruit_ctx = any(tok in name_ascii for tok in fruit_tokens)
    has_person_ctx = any(tok in name_ascii for tok in person_tokens)
    has_tree_ctx = any(tok in name_ascii for tok in tree_tokens)
    if not has_fruit_ctx and not has_person_ctx and not has_tree_ctx:
        return dets

    tree_n = int(sum(1 for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'tree'))
    dumbbell_n = int(sum(1 for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'dumbbell'))
    flower_n = int(sum(1 for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower'))
    fruit_n = int(
        sum(
            1
            for d in dets
            if (normalize_class_name(d.get('class', '')) or '') in ({'fruit'} | set(SPECIFIC_FRUIT_CLASSES))
        )
    )

    filtered = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or ''
        confv = float(d.get('conf', 0.0))

        # Fruit filename context: strongly suppress non-fruit noise.
        if has_fruit_ctx:
            if cls_name == 'flower':
                if fruit_n >= 2:
                    continue
                if fruit_n >= 1 and confv < 0.97:
                    continue
                if fruit_n == 0 and confv < 0.96:
                    continue
            if cls_name == 'tree':
                if fruit_n == 0 and confv < 0.90:
                    continue
                if fruit_n > 0 and confv < 0.82:
                    continue

        # Person filename context: drop face/body-like flower boxes unless very strong and small.
        if has_person_ctx and cls_name == 'flower':
            if confv < 0.85:
                continue

        # Tree filename context: suppress weak flower noise when tree is present.
        if has_tree_ctx and cls_name == 'flower' and tree_n >= 1 and fruit_n == 0:
            if confv < 0.88:
                continue

        if cls_name != 'tree':
            filtered.append(d)
            continue

        # Preserve original tree guard for fruit-context images only.
        if has_fruit_ctx and tree_n > 0 and dumbbell_n == 0:
            if fruit_n == 0 and confv < 0.90:
                continue
            if fruit_n > 0 and confv < 0.82:
                continue
        if has_person_ctx and confv < 0.50 and flower_n >= 1:
            continue
        filtered.append(d)
    return canonicalize_final_detections(filtered)


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


def suppress_scene_level_boxes(detections: list, img_w: int, img_h: int) -> list:
    """Suppress oversized scene-level boxes when object-level boxes already exist.
    This prevents "scan entire image" detections from inflating counts.
    """
    # PATCH: reduce flower FP / fruit confusion
    # Accept optional image region checks when an image numpy array is available
    # (function signature kept backward-compatible; callers may pass image_np
    # as a keyword arg to enable color/petal checks).
    # Note: image_np is not part of signature for backward compatibility
    # but may be passed in via kwargs by callers in this file.
    # If image_np not provided, function behaves as before (geometry-only rules).
    extra_image = None
    # allow callers to pass image via kwarg 'image_np' (keeps existing call sites valid)
    try:
        # hack: inspect caller-supplied variable through closure of arguments
        # but since apply_patch can't change all call sites reliably, we instead
        # detect if a global variable 'CURRENT_IMAGE_FOR_SCENE_CHECK' exists.
        extra_image = globals().get('CURRENT_IMAGE_FOR_SCENE_CHECK', None)
    except Exception:
        extra_image = None

    if not detections or img_w <= 0 or img_h <= 0:
        return canonicalize_final_detections(detections)

    dets = canonicalize_final_detections(detections)
    small_per_class = {}
    counts = {}
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        counts[cls_name] = counts.get(cls_name, 0) + 1
        confv = float(d.get('conf', 0.0))
        area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if 0 < area_ratio <= 0.14 and confv >= 0.35:
            small_per_class[cls_name] = small_per_class.get(cls_name, 0) + 1

    out = []
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        confv = float(d.get('conf', 0.0))
        area_ratio, _, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)

        # If class already has local object boxes, drop giant class-level boxes.
        if small_per_class.get(cls_name, 0) >= 2:
            if near_full:
                continue
            if area_ratio >= 0.48 and confv < 0.95:
                continue

        # Flower-specific guard: avoid giant flower boxes that swallow many flowers.
        if cls_name == 'flower':
            flower_n = int(counts.get('flower', 0))
            # DROP large flower boxes that are green-dominant with little petal evidence.
            if area_ratio >= 0.30 and extra_image is not None:
                try:
                    x1, y1, x2, y2 = map(int, d.get('xyxy')[:4])
                    x1 = max(0, min(img_w - 1, x1))
                    x2 = max(0, min(img_w, x2))
                    y1 = max(0, min(img_h - 1, y1))
                    y2 = max(0, min(img_h, y2))
                    if x2 > x1 and y2 > y1:
                        roi = extra_image[y1:y2, x1:x2]
                        if roi.size > 0:
                            try:
                                hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                                hch = hsv[:, :, 0]
                                sch = hsv[:, :, 1]
                                vch = hsv[:, :, 2]
                                non_green = ((hch < 35) | (hch > 95))
                                colorful = (sch >= 55) & (vch >= 40)
                                bright_white = (vch >= 170) & (sch <= 55)
                                petal_like = (non_green & colorful) | bright_white
                                green_like = ((hch >= 35) & (hch <= 95) & (sch >= 40) & (vch >= 35))
                                petal_ratio = float(np.count_nonzero(petal_like)) / float(max(1, roi.shape[0] * roi.shape[1]))
                                green_ratio = float(np.count_nonzero(green_like)) / float(max(1, roi.shape[0] * roi.shape[1]))
                                if green_ratio > 0.50 and petal_ratio < 0.06:
                                    continue
                            except Exception:
                                pass
                except Exception:
                    pass

            if flower_n <= 2:
                # Sparse mode: reject broad weak boxes that touch image border (scene-scan style).
                border_touch = False
                try:
                    x1, y1, x2, y2 = map(float, d.get('xyxy')[:4])
                    border_touch = (x1 <= 0.02 * img_w) or (y1 <= 0.02 * img_h) or (x2 >= 0.98 * img_w) or (y2 >= 0.98 * img_h)
                except Exception:
                    border_touch = near_full
                if near_full and confv < 0.92:
                    continue
                if border_touch and area_ratio >= 0.30 and confv < 0.88:
                    continue
                if border_touch and area_ratio >= 0.20 and confv < 0.76:
                    continue
                if area_ratio >= 0.85 and confv < 0.82:
                    continue
            else:
                if area_ratio >= 0.60:
                    continue
                if area_ratio >= 0.30 and confv < 0.90:
                    continue
                if small_per_class.get('flower', 0) >= 1 and area_ratio >= 0.35:
                    continue

        if cls_name == 'tree':
            # Tree scene-box suppression: keep local canopies, drop broad weak boxes.
            if area_ratio >= 0.45 and confv < 0.60:
                continue
            if area_ratio >= 0.25 and confv < 0.42:
                continue

        if cls_name == 'fruit' or cls_name in SPECIFIC_FRUIT_CLASSES:
            # Fruit should be object-level; broad weak fruit boxes are usually wrong.
            if area_ratio >= 0.25 and confv < 0.68:
                continue

        # Extremely large boxes are usually scene scans and should be removed,
        # except a very high-confidence single-object close-up.
        if area_ratio >= 0.88 and confv < 0.96:
            continue

        out.append(d)

    # Safety fallback: if suppression removed everything, return original dets.
    return out if out else dets


def normalize_dense_flower_boxes(detections: list, img_w: int, img_h: int) -> list:
    """Normalize flower boxes in dense scenes to avoid messy oversized boxes.

    Steps:
    - remove parent-like flower boxes that cover multiple smaller flowers
    - suppress abnormal large/elongated flower boxes in dense scenes
    - tighten remaining flower boxes toward object center for cleaner overlays
    """
    dets = canonicalize_final_detections(detections)
    if not dets or img_w <= 0 or img_h <= 0:
        return dets

    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    if len(flowers) < 3:
        return dets

    areas = []
    for d in flowers:
        a, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if a > 0:
            areas.append(a)
    if not areas:
        return dets
    med_area = sorted(areas)[len(areas) // 2]

    dense_mode = len(flowers) >= 6
    ordered = sorted(flowers, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    kept = []
    for d in ordered:
        confv = float(d.get('conf', 0.0))
        area_ratio, aspect, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)
        if area_ratio <= 0:
            continue
        if near_full:
            continue

        if dense_mode:
            max_dense_area = min(0.22, max(0.08, 3.2 * float(med_area)))
            if area_ratio > max_dense_area and confv < 0.96:
                continue
            if aspect > 2.8 and confv < 0.95:
                continue

        try:
            x1, y1, x2, y2 = map(float, d.get('xyxy')[:4])
        except Exception:
            continue
        d_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if d_area <= 0:
            continue

        child_n = 0
        child_top_conf = 0.0
        for o in ordered:
            if o is d:
                continue
            o_area_ratio, _, _ = _box_metrics(o.get('xyxy'), img_w, img_h)
            if o_area_ratio <= 0:
                continue
            try:
                ox1, oy1, ox2, oy2 = map(float, o.get('xyxy')[:4])
            except Exception:
                continue
            o_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
            if o_area <= 0 or o_area > 0.62 * d_area:
                continue
            ocx = (ox1 + ox2) * 0.5
            ocy = (oy1 + oy2) * 0.5
            if not (x1 <= ocx <= x2 and y1 <= ocy <= y2):
                continue
            if _box_iou(d.get('xyxy'), o.get('xyxy')) < 0.08:
                continue
            child_n += 1
            child_top_conf = max(child_top_conf, float(o.get('conf', 0.0)))

        if child_n >= 2 and confv <= child_top_conf + 0.08:
            continue

        kept.append(dict(d))

    if not kept:
        return dets

    # Guardrail: avoid over-pruning in difficult scenes.
    if len(kept) < max(2, int(round(len(flowers) * 0.45))):
        kept = [dict(d) for d in flowers]

    # Tighten flower boxes around center so visualization is less noisy.
    tightened = []
    tighten_enabled = len(kept) >= 4
    for d in kept:
        xy = d.get('xyxy') or []
        if len(xy) < 4:
            continue
        try:
            x1, y1, x2, y2 = map(float, xy[:4])
        except Exception:
            continue
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 0 or bh <= 0:
            continue

        if tighten_enabled:
            area_ratio, _, _ = _box_metrics([x1, y1, x2, y2], img_w, img_h)
            shrink = 0.86
            if area_ratio >= 0.08:
                shrink = 0.80
            elif area_ratio <= 0.010:
                shrink = 0.90
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            nw = max(12.0, bw * shrink)
            nh = max(12.0, bh * shrink)
            x1 = max(0.0, cx - 0.5 * nw)
            y1 = max(0.0, cy - 0.5 * nh)
            x2 = min(float(img_w - 1), cx + 0.5 * nw)
            y2 = min(float(img_h - 1), cy + 0.5 * nh)

        d2 = dict(d)
        d2['xyxy'] = [x1, y1, x2, y2]
        tightened.append(d2)

    non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    merged = non_flower + tightened
    merged = dedup_detections_by_class_nms_classwise(merged, default_iou=0.72)
    return merged


def _box_center_and_diag(xyxy):
    if not xyxy or len(xyxy) < 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, xyxy[:4])
    except Exception:
        return None
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return None
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    diag = math.hypot(w, h)
    return cx, cy, diag


def cluster_flower_detections_by_center(detections: list) -> list:
    """Cluster flower detections by center proximity + overlap and keep one rep per object.
    This stabilizes counting when multiple boxes are produced for one flower.
    """
    flower_dets = [
        d for d in canonicalize_final_detections(detections)
        if (normalize_class_name(d.get('class', '')) or '') == 'flower'
    ]
    if not flower_dets:
        return []

    ordered = sorted(flower_dets, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
    clusters = []
    for d in ordered:
        geom = _box_center_and_diag(d.get('xyxy'))
        if geom is None:
            continue
        cx, cy, diag = geom
        assigned = False
        for c in clusters:
            rep = c['rep']
            rep_geom = _box_center_and_diag(rep.get('xyxy'))
            if rep_geom is None:
                continue
            rcx, rcy, rdiag = rep_geom
            dist = math.hypot(cx - rcx, cy - rcy)
            iou = _box_iou(d.get('xyxy'), rep.get('xyxy'))
            # Keep center gate tight; nearby different flowers should not collapse.
            center_gate = max(8.0, 0.10 * min(diag, rdiag))
            # Merge only with strong overlap OR very-close centers with overlap evidence.
            if iou >= 0.66 or (dist <= center_gate and iou >= 0.18):
                c['items'].append(d)
                if float(d.get('conf', 0.0)) > float(rep.get('conf', 0.0)):
                    c['rep'] = d
                assigned = True
                break
        if not assigned:
            clusters.append({'rep': d, 'items': [d]})

    out = []
    for c in clusters:
        rep = dict(c['rep'])
        rep['class'] = 'flower'
        out.append(rep)
    return out

def recover_flower_instances(
    model,
    image_np: np.ndarray,
    base_filtered: list,
    tuned_conf: float,
    imgsz: int,
    use_fp16: bool,
):
    """Recover missed flowers using a permissive dedicated pass + tiled pass.
    Returns updated full detection list.
    """
    if image_np is None or image_np.size == 0:
        return canonicalize_final_detections(base_filtered)

    dets = canonicalize_final_detections(base_filtered)
    flowers_now = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']

    # Trigger only when flower count is likely under-estimated.
    h, w = image_np.shape[:2]
    flower_areas = []
    for d in flowers_now:
        area_ratio, _, _ = _box_metrics(d.get('xyxy'), w, h)
        if area_ratio > 0:
            flower_areas.append(area_ratio)
    flower_areas.sort()
    med_area = flower_areas[len(flower_areas) // 2] if flower_areas else 0.0
    dense_small = (med_area > 0.0 and med_area < 0.002)
    max_flowers_before_recover = 18 if dense_small else 6
    if len(flowers_now) > max_flowers_before_recover:
        return dets

    # Pass 1: permissive full-image flower extraction.
    low_conf = max(0.08 if dense_small else 0.10, float(tuned_conf) * (0.38 if dense_small else 0.42))
    hi_imgsz = int(max(int(imgsz), 960))
    if dense_small:
        hi_imgsz = int(max(hi_imgsz, 1152))
    _, _, low_dets = run_detection_on_image(
        model,
        image_np,
        conf=low_conf,
        imgsz=hi_imgsz,
        use_fp16=use_fp16,
        nms_iou=0.84,
        max_det=640,
    )
    low_flowers = []
    h, w = image_np.shape[:2]
    for d in canonicalize_final_detections(low_dets):
        if (normalize_class_name(d.get('class', '')) or '') != 'flower':
            continue
        confv = float(d.get('conf', 0.0))
        area_ratio, _, near_full = _box_metrics(d.get('xyxy'), w, h)
        if confv < max(0.18, low_conf):
            continue
        if area_ratio <= 0:
            continue
        if area_ratio < 0.00008 and confv < 0.42:
            continue
        if near_full and confv < 0.92:
            continue
        low_flowers.append(d)

    # Pass 2: tiled flower recovery.
    tiled = run_tiled_flower_detections(
        model,
        image_np,
        base_conf=max(0.10, float(tuned_conf) * 0.50),
        imgsz=hi_imgsz,
        use_fp16=use_fp16,
    )

    candidate_flowers = flowers_now + low_flowers + tiled
    if not candidate_flowers:
        return dets

    # Visual evidence refinement to reduce false positives before clustering.
    if dense_small or len(flowers_now) >= 2:
        try:
            candidate_flowers = refine_flower_boxes_with_visual_evidence(image_np, candidate_flowers, img_w=w, img_h=h)
        except Exception:
            candidate_flowers = candidate_flowers
    if not candidate_flowers:
        return dets

    clustered = cluster_flower_detections_by_center(candidate_flowers)
    if not clustered:
        return dets

    # Guard against runaway counts when the base pass saw zero flowers.
    if len(flowers_now) == 0:
        avg_conf = (sum(float(d.get('conf', 0.0)) for d in clustered) / len(clustered)) if clustered else 0.0
        max_recovered = 28 if dense_small else 14
        if avg_conf < 0.45 and len(clustered) > max_recovered:
            clustered = sorted(clustered, key=lambda x: float(x.get('conf', 0.0)), reverse=True)[:max_recovered]

    if len(clustered) <= len(flowers_now):
        return dets
    non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    return non_flower + clustered


def recount_flowers_strict(
    flower_model,
    image_np: np.ndarray,
    base_detections: list,
    base_conf: float,
    imgsz: int,
    use_fp16: bool,
):
    """Final flower-only recount for static images."""
    if image_np is None or image_np.size == 0:
        return canonicalize_final_detections(base_detections)
    if flower_model is None:
        return canonicalize_final_detections(base_detections)

    h, w = image_np.shape[:2]
    if h <= 0 or w <= 0:
        return canonicalize_final_detections(base_detections)

    seed = canonicalize_final_detections(base_detections)
    seed_flowers = [d for d in seed if (normalize_class_name(d.get('class', '')) or '') == 'flower']

    conf_a = max(0.10, float(base_conf) * 0.45)
    conf_b = max(0.09, float(base_conf) * 0.38)

    _, _, dets_a = run_detection_on_image(
        flower_model,
        image_np,
        conf=conf_a,
        imgsz=max(int(imgsz), 960),
        use_fp16=use_fp16,
        nms_iou=0.80,
        max_det=360,
    )
    _, _, dets_b = run_detection_on_image(
        flower_model,
        image_np,
        conf=conf_b,
        imgsz=max(int(imgsz), 1152),
        use_fp16=use_fp16,
        nms_iou=0.82,
        max_det=420,
    )
    tiled = run_tiled_flower_detections(
        flower_model,
        image_np,
        base_conf=max(0.10, conf_a),
        imgsz=max(int(imgsz), 1024),
        use_fp16=use_fp16,
    )

    candidates = []
    for d in (seed_flowers + dets_a + dets_b + tiled):
        d2 = dict(d)
        d2['class'] = 'flower'
        confv = float(d2.get('conf', 0.0))
        area_ratio, _, near_full = _box_metrics(d2.get('xyxy'), w, h)
        if area_ratio <= 0:
            continue
        if confv < max(0.12, float(base_conf) * 0.35):
            continue
        if area_ratio < 0.00005 and confv < 0.50:
            continue
        if near_full:
            continue
        if area_ratio >= 0.40 and confv < 0.94:
            continue
        candidates.append(d2)

    if not candidates:
        return seed

    flowers = dedup_detections_by_class_nms_classwise(candidates, default_iou=0.94)
    flowers = suppress_scene_level_boxes(flowers, img_w=w, img_h=h)
    # In dense fields, avoid aggressive one-object merging.
    flowers = dedup_detections_by_class_nms_classwise(flowers, default_iou=0.90)
    flowers = collapse_sparse_flower_duplicates(flowers, img_w=w, img_h=h)

    non_flower = [d for d in seed if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    return non_flower + flowers


def enforce_one_object_one_box(detections: list, img_w: int, img_h: int) -> list:
    """Enforce one object ~= one box by removing scene boxes and merging duplicates.

    This pass is intentionally class-aware and conservative:
    - drop near-full / oversized boxes that usually represent scene scans
    - merge same-object duplicates using IoU, IoM and center-distance
    - remove parent boxes that cover multiple smaller kept boxes of same class
    """
    dets = canonicalize_final_detections(detections)
    if not dets or img_w <= 0 or img_h <= 0:
        return dets

    grouped = {}
    for d in dets:
        cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
        d2 = dict(d)
        d2['class'] = cls_name
        grouped.setdefault(cls_name, []).append(d2)

    out = []
    for cls_name, cls_dets in grouped.items():
        # Only treat extremely small flower sets as sparse. With 3-4 flowers,
        # center-based merge can undercount in overlapping scenes.
        sparse_flower_mode = (cls_name == 'flower' and len(cls_dets) <= 2)
        dense_tree_mode = (cls_name == 'tree' and len(cls_dets) >= 3)
        ordered = sorted(cls_dets, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
        kept = []
        for d in ordered:
            confv = float(d.get('conf', 0.0))
            area_ratio, _, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)
            if area_ratio <= 0:
                continue

            # reject global scene boxes unless extremely confident
            if near_full:
                if cls_name == 'flower' and sparse_flower_mode:
                    if confv < 0.93:
                        continue
                elif confv < 0.97:
                    continue
            if area_ratio >= 0.72:
                if cls_name == 'flower' and sparse_flower_mode:
                    if area_ratio >= 0.85 and confv < 0.88:
                        continue
                elif confv < 0.96:
                    continue

            geom_d = _box_center_and_diag(d.get('xyxy'))
            if geom_d is None:
                continue
            cx_d, cy_d, diag_d = geom_d

            duplicated = False
            for k in kept:
                iou = _box_iou(d.get('xyxy'), k.get('xyxy'))
                iou_dup_thr = 0.52 if sparse_flower_mode else 0.62
                if iou >= iou_dup_thr:
                    duplicated = True
                    break

                # IoM for size-mismatch duplicate boxes
                try:
                    dx1, dy1, dx2, dy2 = map(float, d.get('xyxy')[:4])
                    kx1, ky1, kx2, ky2 = map(float, k.get('xyxy')[:4])
                    inter_x1 = max(dx1, kx1)
                    inter_y1 = max(dy1, ky1)
                    inter_x2 = min(dx2, kx2)
                    inter_y2 = min(dy2, ky2)
                    inter_w = max(0.0, inter_x2 - inter_x1)
                    inter_h = max(0.0, inter_y2 - inter_y1)
                    inter = inter_w * inter_h
                    d_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
                    k_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
                    min_area = min(max(1e-12, d_area), max(1e-12, k_area))
                    iom = inter / min_area if min_area > 0 else 0.0
                except Exception:
                    iom = 0.0
                iom_dup_thr = 0.58 if sparse_flower_mode else 0.72
                if iom >= iom_dup_thr:
                    duplicated = True
                    break

                geom_k = _box_center_and_diag(k.get('xyxy'))
                if geom_k is None:
                    continue
                cx_k, cy_k, diag_k = geom_k
                dist = math.hypot(cx_d - cx_k, cy_d - cy_k)
                center_thr = (0.14 if sparse_flower_mode else 0.15) * min(diag_d, diag_k)
                if sparse_flower_mode:
                    # In sparse mode require some overlap evidence before merging by center distance.
                    if dist <= max(8.0, center_thr) and (iou >= 0.12 or iom >= 0.30):
                        duplicated = True
                        break
                else:
                    if cls_name == 'flower':
                        # Dense flowers overlap naturally; merge only with strong duplicate evidence.
                        if dist <= max(6.0, 0.10 * min(diag_d, diag_k)) and (iou >= 0.45 or iom >= 0.72):
                            duplicated = True
                            break
                    elif cls_name == 'tree':
                        # In dense tree scenes, be more conservative to avoid collapsing nearby true trees.
                        if dense_tree_mode:
                            if dist <= max(8.0, center_thr) and (iou >= 0.45 or iom >= 0.65):
                                duplicated = True
                                break
                        else:
                            # Tree can appear in tight groups; require overlap evidence before center-merge.
                            if dist <= max(8.0, center_thr) and (iou >= 0.35 or iom >= 0.55):
                                duplicated = True
                                break
                    else:
                        if dist <= max(8.0, center_thr):
                            duplicated = True
                            break

                # Extra suppression for near-contained boxes in sparse flower scenes.
                # This targets 2 overlapping boxes over 1 flower (same center, size jitter).
                if sparse_flower_mode and iou >= 0.34 and iom >= 0.50 and dist <= max(10.0, 0.30 * min(diag_d, diag_k)):
                    duplicated = True
                    break

            if not duplicated:
                kept.append(d)

        # remove parent boxes that cover multiple same-class kept boxes
        final_cls = []
        for i, k in enumerate(kept):
            if dense_tree_mode:
                # Keep dense tree candidates; parent suppression here can over-collapse clustered trees.
                pass
            k_conf = float(k.get('conf', 0.0))
            k_area, _, _ = _box_metrics(k.get('xyxy'), img_w, img_h)
            covered_children = 0
            max_child_conf = 0.0
            max_child_area = 0.0
            for j, o in enumerate(kept):
                if i == j:
                    continue
                o_area, _, _ = _box_metrics(o.get('xyxy'), img_w, img_h)
                if o_area <= 0:
                    continue
                iou_ko = _box_iou(k.get('xyxy'), o.get('xyxy'))
                # parent-like relation: k is much larger and overlaps child
                parent_overlap = iou_ko >= 0.18
                try:
                    kx1, ky1, kx2, ky2 = map(float, k.get('xyxy')[:4])
                    ox1, oy1, ox2, oy2 = map(float, o.get('xyxy')[:4])
                    ocx = 0.5 * (ox1 + ox2)
                    ocy = 0.5 * (oy1 + oy2)
                    # Sparse flower/tree scenes: allow center-in-parent as overlap evidence.
                    sparse_parent_mode_local = (len(kept) <= 4 and cls_name in {'flower', 'tree'})
                    if (not parent_overlap) and sparse_parent_mode_local and (kx1 <= ocx <= kx2) and (ky1 <= ocy <= ky2):
                        parent_overlap = True
                except Exception:
                    pass
                if k_area > 0 and (o_area <= 0.62 * k_area) and parent_overlap:
                    covered_children += 1
                    max_child_conf = max(max_child_conf, float(o.get('conf', 0.0)))
                    max_child_area = max(max_child_area, o_area)

            sparse_parent_mode = (len(kept) <= 4 and cls_name in {'flower', 'tree'})
            if sparse_parent_mode and max_child_area > 0:
                if (
                    covered_children >= 1
                    and k_area >= (1.45 * max_child_area)
                    and k_conf <= max_child_conf + 0.28
                ):
                    continue

            if covered_children >= 2 and k_conf <= max_child_conf + (0.12 if cls_name in {'flower', 'tree'} else 0.08):
                continue
            final_cls.append(k)

        out.extend(final_cls)

    return out


def trim_sparse_flower_outliers(detections: list, img_w: int, img_h: int) -> list:
    """Trim false flower outliers in sparse/single-flower scenes.

    Apply only when flower count is small and boxes are relatively large,
    which usually indicates close-up photos where duplicate/noise boxes appear.
    """
    dets = canonicalize_final_detections(detections)
    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    if not flowers:
        return dets

    n = len(flowers)
    if n < 2 or n > 6:
        return dets

    areas = []
    for d in flowers:
        a, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        if a > 0:
            areas.append(a)
    if not areas:
        return dets

    areas_sorted = sorted(areas)
    med_area = areas_sorted[len(areas_sorted) // 2]
    # only sparse scene: close-up flowers (not dense fields)
    if med_area < 0.035:
        return dets

    best = max(flowers, key=lambda d: float(d.get('conf', 0.0)))
    best_conf = float(best.get('conf', 0.0))
    best_geom = _box_center_and_diag(best.get('xyxy'))
    if best_geom is None:
        return dets
    bcx, bcy, bdiag = best_geom

    kept_flower = []
    for d in flowers:
        confv = float(d.get('conf', 0.0))
        area_ratio, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
        geom = _box_center_and_diag(d.get('xyxy'))
        if geom is None:
            continue
        cx, cy, _ = geom
        dist = math.hypot(cx - bcx, cy - bcy)
        iou_best = _box_iou(d.get('xyxy'), best.get('xyxy'))

        # Keep confident boxes that are not extreme outliers by area/location.
        if confv < max(0.45, best_conf - 0.28):
            continue
        if n <= 3 and best_conf >= 0.88 and confv < (best_conf - 0.18) and iou_best < 0.20:
            continue
        if area_ratio < 0.006 or area_ratio > 0.48:
            continue
        if (iou_best < 0.08) and (dist > (0.75 * bdiag)) and confv < 0.88:
            continue
        kept_flower.append(d)

    # Collapse any remaining duplicates among sparse flowers.
    kept_flower = cluster_flower_detections_by_center(kept_flower)
    if not kept_flower:
        return dets

    non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    return non_flower + kept_flower


def trim_sparse_custom_outliers(detections: list, img_w: int, img_h: int) -> list:
    """Generic sparse-scene outlier trimming for all custom classes.

    Helps remove oversized parent boxes or weak stray boxes so each object tends to
    be represented by one box, across flower/fruit/tree/dumbbell.
    """
    dets = canonicalize_final_detections(detections)
    if not dets or img_w <= 0 or img_h <= 0:
        return dets

    custom_classes = get_custom_class_names()
    grouped = {}
    for d in dets:
        c = normalize_class_name(d.get('class', '')) or 'unknown'
        grouped.setdefault(c, []).append(d)

    out = []
    for cls_name, cls_dets in grouped.items():
        if cls_name not in custom_classes or len(cls_dets) < 2:
            out.extend(cls_dets)
            continue

        # Preserve dense scenes for this class to avoid undercount.
        if len(cls_dets) > 8:
            out.extend(cls_dets)
            continue

        areas = []
        for d in cls_dets:
            a, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
            if a > 0:
                areas.append(a)
        if not areas:
            out.extend(cls_dets)
            continue

        med_area = sorted(areas)[len(areas) // 2]
        top_conf = max(float(d.get('conf', 0.0)) for d in cls_dets)

        kept = []
        for d in cls_dets:
            confv = float(d.get('conf', 0.0))
            d_area, _, _ = _box_metrics(d.get('xyxy'), img_w, img_h)
            if d_area <= 0:
                continue

            # parent-box detector: box covers several smaller same-class boxes
            child_n = 0
            child_top_conf = 0.0
            for o in cls_dets:
                if o is d:
                    continue
                o_area, _, _ = _box_metrics(o.get('xyxy'), img_w, img_h)
                if o_area <= 0:
                    continue
                iou_do = _box_iou(d.get('xyxy'), o.get('xyxy'))
                geom_o = _box_center_and_diag(o.get('xyxy'))
                if geom_o is None:
                    continue
                ox, oy, _ = geom_o
                try:
                    x1, y1, x2, y2 = map(float, d.get('xyxy')[:4])
                except Exception:
                    continue
                center_inside = (x1 <= ox <= x2) and (y1 <= oy <= y2)
                if center_inside and o_area <= 0.65 * d_area and iou_do >= 0.05:
                    child_n += 1
                    child_top_conf = max(child_top_conf, float(o.get('conf', 0.0)))

            if child_n >= 1 and d_area > (2.4 * med_area) and confv <= child_top_conf + 0.06:
                continue

            # sparse-scene outlier gates
            if len(cls_dets) <= 4 and med_area >= 0.02:
                if d_area > (2.8 * med_area) and confv < top_conf + 0.02:
                    continue
                if d_area < (0.15 * med_area) and confv < max(0.35, top_conf - 0.20):
                    continue

            kept.append(d)

        kept = kept if kept else cls_dets

        # Sparse-scene duplicate collapse:
        # when this class appears only a few times with relatively large boxes,
        # multiple overlapping boxes are usually the same object.
        if len(kept) <= 6 and med_area >= 0.015 and cls_name != 'tree':
            ordered = sorted(kept, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
            clusters = []
            for d in ordered:
                g_d = _box_center_and_diag(d.get('xyxy'))
                if g_d is None:
                    continue
                cx_d, cy_d, diag_d = g_d
                merged = False
                for c in clusters:
                    rep = c['rep']
                    g_r = _box_center_and_diag(rep.get('xyxy'))
                    if g_r is None:
                        continue
                    cx_r, cy_r, diag_r = g_r
                    iou = _box_iou(d.get('xyxy'), rep.get('xyxy'))
                    dist = math.hypot(cx_d - cx_r, cy_d - cy_r)
                    if iou >= 0.20 or dist <= (0.28 * min(diag_d, diag_r)):
                        c['items'].append(d)
                        if float(d.get('conf', 0.0)) > float(rep.get('conf', 0.0)):
                            c['rep'] = d
                        merged = True
                        break
                if not merged:
                    clusters.append({'rep': d, 'items': [d]})

            collapsed = []
            for c in clusters:
                rep = dict(c['rep'])
                rep['class'] = cls_name
                collapsed.append(rep)
            kept = collapsed if collapsed else kept

        out.extend(kept)

    return out


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
    elif cls_name == 'fruit' or cls_name in SPECIFIC_FRUIT_CLASSES:
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
        if cls_name != 'fruit' and cls_name not in SPECIFIC_FRUIT_CLASSES:
            continue
        best = None
        best_score = -1.0
        for c in coco_fruits:
            coco_conf = float(c.get('conf', 0.0))
            # allow COCO fruit to influence mapping when it has stronger evidence
            d_conf = float(d.get('conf', 0.0))
            min_coco_conf = (0.70 if cls_name == 'fruit' else 0.65)
            # require coco_conf to be at least a reasonable floor or stronger than custom
            if coco_conf < max(min_coco_conf, d_conf - 0.05):
                continue
            iou = _box_iou(d.get('xyxy'), c.get('xyxy'))
            min_iou = (0.60 if cls_name == 'fruit' else 0.45)
            if iou < min_iou:
                continue
            # score favors both high IoU and high coco confidence
            score = iou * coco_conf
            if score > best_score:
                best_score = score
                best = c
        if best is not None:
            coco_conf = float(best.get('conf', 0.0))
            d_conf = float(d.get('conf', 0.0))
            # If COCO is clearly stronger, map the custom detection to the specific COCO fruit.
            if coco_conf >= d_conf + 0.03 or cls_name == 'fruit':
                d['class'] = str(best.get('class', 'fruit')).lower().strip()
                d['_refined_from'] = 'fruit'
                d['_refined_by'] = 'coco'
                d['_coco_conf'] = coco_conf
            else:
                # If evidence is similar, annotate as coincident without overwriting label so
                # downstream hybrid logic can choose the best representation.
                d.setdefault('_coincident_coco', str(best.get('class', '')).lower().strip())
                d['_coco_conf'] = coco_conf


def rescue_fruit_from_coco_when_flower_only(merged_dets: list, coco_dets: list, img_w: int, img_h: int, base_conf: float) -> list:
    """Replace flower-only outputs with strong overlapping COCO fruit detections."""
    dets = canonicalize_final_detections(merged_dets)
    if not dets:
        return dets

    classes = set()
    for d in dets:
        classes.add(normalize_class_name(d.get('class', '')) or 'unknown')

    if classes != {'flower'}:
        return dets

    flowers = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') == 'flower']
    if not flowers or len(flowers) > 3:
        return dets

    coco_names = set(SPECIFIC_FRUIT_CLASSES)
    coco_fruits = []
    for c in (coco_dets or []):
        cname = normalize_class_name(c.get('class', '')) or 'unknown'
        if cname not in coco_names:
            continue
        confv = float(c.get('conf', 0.0))
        if confv < max(0.42, float(base_conf) + 0.04):
            continue
        area_ratio, _, _ = _box_metrics(c.get('xyxy'), img_w, img_h)
        if area_ratio <= 0:
            continue
        c2 = dict(c)
        c2['class'] = cname
        c2['_source_model'] = 'coco_fruit_rescue'
        coco_fruits.append(c2)

    if not coco_fruits:
        return dets

    kept_flowers = []
    rescued = []
    used_fruit_ids = set()
    for f in flowers:
        best = None
        best_score = -1.0
        for c in coco_fruits:
            cid = id(c)
            if cid in used_fruit_ids:
                continue
            iou = _box_iou(f.get('xyxy'), c.get('xyxy'))
            if iou < 0.22:
                continue
            score = iou * float(c.get('conf', 0.0))
            if score > best_score:
                best_score = score
                best = c
        if best is not None and float(best.get('conf', 0.0)) >= max(0.42, float(base_conf) + 0.04):
            used_fruit_ids.add(id(best))
            rescued.append(best)
        else:
            kept_flowers.append(f)

    if not rescued:
        return dets

    non_flower = [d for d in dets if (normalize_class_name(d.get('class', '')) or '') != 'flower']
    merged = non_flower + kept_flowers + rescued
    return dedup_detections_by_class_nms_classwise(merged, default_iou=0.60)


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
        'flower': 0.70,
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
        # additional IoM (intersection over smaller box area) threshold helps
        # collapse many small overlaps produced by tiled inference where IoU
        # can be artificially low due to size differences.
        iom_threshold = 0.60 if cls_name == 'flower' else 0.55
        if cls_name == 'flower' and len(dets) >= 10:
            areas = []
            for dd in dets:
                xy = dd.get('xyxy') or []
                try:
                    x1, y1, x2, y2 = map(float, xy[:4])
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                except Exception:
                    area = 0.0
                if area > 0:
                    areas.append(area)
            if areas:
                areas.sort()
                med_area = areas[len(areas) // 2]
                max_area = max(areas)
                # Dense tiny flowers: keep more overlapping boxes to avoid undercount.
                if max_area > 0 and (med_area / max_area) < 0.10:
                    iou_threshold = max(iou_threshold, 0.72)
                    iom_threshold = max(iom_threshold, 0.62)

        ordered = sorted(dets, key=lambda x: float(x.get('conf', 0.0)), reverse=True)
        kept = []
        for d in ordered:
            overlapped = False
            dax = d.get('xyxy') or []
            # compute area of d once
            try:
                dax1, day1, dax2, day2 = map(float, dax[:4])
                d_area = max(0.0, dax2 - dax1) * max(0.0, day2 - day1)
            except Exception:
                d_area = 0.0

            for k in kept:
                kax = k.get('xyxy') or []
                # IoU check (existing behavior)
                iou = _box_iou(dax, kax)
                if iou >= iou_threshold:
                    overlapped = True
                    break

                # Intersection-over-min-area check: if the intersection covers a large
                # fraction of the smaller box, treat as duplicate as well.
                try:
                    kx1, ky1, kx2, ky2 = map(float, kax[:4])
                    k_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
                    inter_x1 = max(dax1, kx1)
                    inter_y1 = max(day1, ky1)
                    inter_x2 = min(dax2, kx2)
                    inter_y2 = min(day2, ky2)
                    inter_w = max(0.0, inter_x2 - inter_x1)
                    inter_h = max(0.0, inter_y2 - inter_y1)
                    inter = inter_w * inter_h
                    min_area = min(max(1e-12, d_area), max(1e-12, k_area))
                    iom = inter / min_area if min_area > 0 else 0.0
                except Exception:
                    iom = 0.0

                if iom >= iom_threshold:
                    overlapped = True
                    break

                # center-distance + size-ratio suppression: if centers are very
                # close and sizes similar, treat as duplicate (common from
                # overlapping tiles/classifier jitter).
                try:
                    dax_cx = (dax1 + dax2) * 0.5
                    dax_cy = (day1 + day2) * 0.5
                    kx1, ky1, kx2, ky2 = map(float, kax[:4])
                    k_cx = (kx1 + kx2) * 0.5
                    k_cy = (ky1 + ky2) * 0.5
                    dx = dax_cx - k_cx
                    dy = dax_cy - k_cy
                    center_dist = math.hypot(dx, dy)
                    # approximate box diagonals
                    d_diag = math.hypot(max(1e-6, dax2 - dax1), max(1e-6, day2 - day1))
                    k_diag = math.hypot(max(1e-6, kx2 - kx1), max(1e-6, ky2 - ky1))
                    min_diag = min(d_diag, k_diag)
                    size_ratio = (d_area / k_area) if k_area > 0 else 1.0
                    # For flowers, avoid over-merging nearby true flowers by requiring
                    # stronger overlap and a tighter center-distance gate.
                    if cls_name == 'flower':
                        if min_diag > 0 and center_dist <= (0.20 * min_diag) and iom >= 0.55 and 0.7 <= size_ratio <= 1.45:
                            overlapped = True
                            break
                    else:
                        # if centers are within 0.45 * min_diag and sizes within 60%..167%, it's duplicate
                        if min_diag > 0 and center_dist <= (0.45 * min_diag) and 0.6 <= size_ratio <= 1.67:
                            overlapped = True
                            break
                except Exception:
                    pass

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
    try:
        list_history.clear()
    except Exception:
        pass


@st.cache_data(show_spinner=False, ttl=10)
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
            min-width: 320px !important;
            max-width: 320px !important;
        }
        button[kind="header"] {
            display: block !important;
        }
        div[data-testid="collapsedControl"] {
            display: block !important;
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
        header {visibility: visible;}
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
        st.markdown("<h1 class='main-header'>SmartFocus AI</h1>", unsafe_allow_html=True)
        # subtitle removed per request
    with col_nav_2:
        # right-hand header info removed per request
        pass

    st.markdown("---")

    if 'control_model_type' not in st.session_state:
        st.session_state['control_model_type'] = DEFAULT_MODEL_TYPE
    if 'control_confidence' not in st.session_state:
        st.session_state['control_confidence'] = 0.25
    if 'control_nms_iou' not in st.session_state:
        st.session_state['control_nms_iou'] = 0.70
    if 'control_max_det' not in st.session_state:
        st.session_state['control_max_det'] = 220
    if 'control_imgsz' not in st.session_state:
        st.session_state['control_imgsz'] = 640

    for _k, _v in {
        'sb_model_type': st.session_state['control_model_type'],
        'sb_confidence': st.session_state['control_confidence'],
        'sb_nms_iou': st.session_state['control_nms_iou'],
        'sb_max_det': st.session_state['control_max_det'],
        'sb_imgsz': st.session_state['control_imgsz'],
    }.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    st.session_state['control_model_type'] = normalize_ui_model_type(st.session_state.get('control_model_type', DEFAULT_MODEL_TYPE))
    st.session_state['sb_model_type'] = normalize_ui_model_type(st.session_state.get('sb_model_type', st.session_state['control_model_type']))

    def _sync_controls_from(prefix: str):
        st.session_state['control_model_type'] = normalize_ui_model_type(st.session_state.get(f'{prefix}_model_type', DEFAULT_MODEL_TYPE))
        st.session_state[f'{prefix}_model_type'] = st.session_state['control_model_type']
        st.session_state['control_confidence'] = float(st.session_state.get(f'{prefix}_confidence', 0.25))
        st.session_state['control_nms_iou'] = float(st.session_state.get(f'{prefix}_nms_iou', 0.70))
        st.session_state['control_max_det'] = int(st.session_state.get(f'{prefix}_max_det', 220))
        st.session_state['control_imgsz'] = int(st.session_state.get(f'{prefix}_imgsz', 640))

    def _sync_sidebar_controls():
        _sync_controls_from('sb')

    with st.sidebar:
        st.markdown("### Cấu hình hệ thống")
        st.write("Thiết lập tham số cho mô hình nhận diện.")

        model_options = [CUSTOM_ONLY_MODEL_TYPE, HYBRID_MODEL_TYPE, COCO_SMALL_MODEL_TYPE, COCO_NANO_MODEL_TYPE]
        st.selectbox(
            "Kiến trúc mô hình",
            model_options,
            key='sb_model_type',
            on_change=_sync_sidebar_controls,
        )
        # (Hidden) custom model metadata and hybrid status - UI elements removed per request
        custom_meta = read_model_metadata()
        custom_model_exists = os.path.exists(CUSTOM_MODEL_PATH)

        st.slider(
            "Ngưỡng tin cậy (Confidence)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key='sb_confidence',
            on_change=_sync_sidebar_controls,
        )

        st.slider(
            "NMS IoU (Non-Max Suppression)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key='sb_nms_iou',
            on_change=_sync_sidebar_controls,
        )

        st.slider(
            "Max detections",
            min_value=50,
            max_value=1000,
            step=10,
            key='sb_max_det',
            on_change=_sync_sidebar_controls,
        )

        st.select_slider("Image size", options=IMG_SIZE_OPTIONS, key='sb_imgsz', on_change=_sync_sidebar_controls)

        # Dense flower and force-dumbbell UI options hidden per request
        st.session_state['enable_dense_flower_fallback'] = bool(ENABLE_DENSE_FLOWER_FALLBACK)
        st.session_state['force_dumbbell_only'] = False
        st.session_state['force_flower_recovery'] = bool(st.session_state.get('force_flower_recovery', False))

        st.markdown("---")
        st.markdown("### Thông tin Node")
        st.info("Trạng thái: Đang hoạt động\n\nNode ID: SF-VN-01")

        if st.button("Làm mới bộ nhớ đệm"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.toast("Đã xóa cache hệ thống!")
            except Exception:
                # ignore cache clear failures but notify
                try:
                    st.toast("Không thể xóa cache.")
                except Exception:
                    pass
    # Keep a single control panel in the sidebar.

    model_type = normalize_ui_model_type(st.session_state.get('sb_model_type', st.session_state.get('control_model_type', DEFAULT_MODEL_TYPE)))
    st.session_state['control_model_type'] = model_type
    confidence = float(st.session_state.get('control_confidence', 0.25))
    nms_iou = float(st.session_state.get('control_nms_iou', 0.70))
    max_det = int(st.session_state.get('control_max_det', 220))
    imgsz = int(st.session_state.get('control_imgsz', 640))

    # persist selected values to session state for inference functions
    st.session_state['confidence'] = float(confidence)
    st.session_state['nms_iou'] = float(nms_iou)
    st.session_state['max_det'] = int(max_det)

    st.markdown("### SMARTFOCUS UI VERSION 2.0 - STREAMLIT")

    st.markdown("<div class='sf-card'>", unsafe_allow_html=True)
    st.markdown("#### Phân tích dữ liệu mới")
    # simplified input mode: only file upload (local-folder option removed)
    input_mode = "Tải một tệp"

    source_name = None
    source_display_name = None
    file_bytes = None
    is_image = True

    if input_mode == "Tải một tệp":
        uploaded_file = st.file_uploader(
            "Kéo và thả tệp tin vào đây (Ảnh hoặc Video)",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        )
        if uploaded_file is not None:
            source_name = uploaded_file.name
            source_display_name = uploaded_file.name
            is_image = (uploaded_file.type or "").startswith("image")
            file_bytes = uploaded_file.read()
    else:
        default_folder = str(st.session_state.get("local_media_folder", ""))
        folder_path = st.text_input(
            "Đường dẫn thư mục trên máy đang chạy app",
            value=default_folder,
            placeholder="Ví dụ: S:/data/traicay2/train/images",
            help="Cách này tránh lỗi khi cố kéo-thả cả thư mục lớn vào trình duyệt.",
        )
        recursive_scan = st.checkbox("Quét cả thư mục con", value=True)
        st.session_state["local_media_folder"] = folder_path

        if folder_path:
            media_files, folder_error = list_local_media_files(folder_path, recursive=recursive_scan)
            if folder_error:
                st.error(folder_error)
            elif not media_files:
                st.warning("Không tìm thấy ảnh/video hợp lệ trong thư mục đã nhập.")
            else:
                st.caption(f"Đã tìm thấy {len(media_files)} tệp hợp lệ trong thư mục.")
                current_index = int(st.session_state.get("local_media_index", 1))
                current_index = max(1, min(current_index, len(media_files)))
                selected_index = st.number_input(
                    "Chọn tệp theo thứ tự",
                    min_value=1,
                    max_value=len(media_files),
                    value=current_index,
                    step=1,
                )
                st.session_state["local_media_index"] = int(selected_index)

                selected_path = media_files[int(selected_index) - 1]
                try:
                    relative_name = str(selected_path.relative_to(Path(folder_path).expanduser()))
                except Exception:
                    relative_name = selected_path.name

                st.caption(f"Tệp đang chọn: {relative_name}")
                source_name = str(selected_path)
                source_display_name = relative_name
                is_image = selected_path.suffix.lower() in IMAGE_FILE_EXTENSIONS
                try:
                    file_bytes = selected_path.read_bytes()
                except Exception as exc:
                    st.error(f"Không thể đọc tệp đã chọn: {exc}")
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

    if file_bytes is not None and source_name is not None:

        model_path_map = {
            COCO_NANO_MODEL_TYPE: "yolo11n.pt",
            COCO_SMALL_MODEL_TYPE: "yolo11s.pt",
            CUSTOM_ONLY_MODEL_TYPE: CUSTOM_MODEL_PATH,
        }

        t0 = datetime.now()
        with st.spinner('Đang xử lý dữ liệu...'):
            try:
                if model_type == HYBRID_MODEL_TYPE:
                    coco_model = load_model(COCO_MODEL_PATH)

                    def _run_coco():
                        return process_uploaded(
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
                            max_det=int(st.session_state.get('max_det', 100)),
                            prefer_recall_custom=False,
                            mode="coco",
                        )

                    if os.path.exists(CUSTOM_MODEL_PATH):
                        custom_model = load_custom_model()

                        def _run_custom():
                            return process_uploaded(
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
                                max_det=int(st.session_state.get('max_det', 100)),
                                prefer_recall_custom=True,
                                mode="custom",
                            )

                        if HYBRID_PARALLEL_INFERENCE:
                            with ThreadPoolExecutor(max_workers=2) as executor:
                                future_coco = executor.submit(_run_coco)
                                future_custom = executor.submit(_run_custom)
                                ann_coco, counts_coco, raw_coco, raw_img = future_coco.result()
                                ann_custom, counts_custom, raw_custom, raw_img_custom = future_custom.result()
                        else:
                            ann_coco, counts_coco, raw_coco, raw_img = _run_coco()
                            ann_custom, counts_custom, raw_custom, raw_img_custom = _run_custom()

                        if raw_img is None:
                            raw_img = raw_img_custom
                        hybrid_stage_stats = {}
                        hybrid_auto_params = {
                            'coco': (raw_coco.get('auto_params', {}) if isinstance(raw_coco, dict) else {}),
                            'custom': (raw_custom.get('auto_params', {}) if isinstance(raw_custom, dict) else {}),
                        }
                        coco_base_conf = float(hybrid_auto_params.get('coco', {}).get('conf', max(0.15, float(confidence))))
                        custom_base_conf = float(hybrid_auto_params.get('custom', {}).get('conf', max(0.15, float(confidence))))
                        hybrid_conf_floor = max(0.10, min(float(confidence), coco_base_conf, custom_base_conf) - 0.05)

                        if isinstance(raw_custom, dict) and isinstance(raw_coco, dict):
                            # Only refine to specific fruits when custom actually defines them.
                            if not custom_prefers_generic_fruit():
                                refine_fruit_labels_with_coco(
                                    raw_custom.get('detections', []),
                                    raw_coco.get('detections', []),
                                )

                        merged_dets = []
                        if isinstance(raw_coco, dict):
                            for d in (raw_coco.get('detections', []) or []):
                                d2 = dict(d)
                                d2['_source_model'] = 'coco'
                                merged_dets.append(d2)
                        if isinstance(raw_custom, dict):
                            for d in (raw_custom.get('detections', []) or []):
                                d2 = dict(d)
                                d2['_source_model'] = 'custom'
                                merged_dets.append(d2)
                        hybrid_stage_stats['merged_raw'] = summarize_detection_stage(merged_dets)

                        # Post-process merged detections to reduce obvious hybrid errors
                        cls_alias = {}
                        cleaned = []
                        img_h, img_w = (raw_img.shape[0], raw_img.shape[1]) if (is_image and raw_img is not None) else (0, 0)
                        for d in merged_dets:
                            cls_name = str(d.get('class', '')).lower().strip()
                            cls_name = cls_alias.get(cls_name, cls_name)
                            d['class'] = cls_name
                            confv = float(d.get('conf', 0.0))
                            if confv < hybrid_conf_floor:
                                if cls_name in {'dumbbell', 'tree'} and confv >= max(0.16, hybrid_conf_floor * 0.70):
                                    pass
                                else:
                                    continue

                            if img_w > 0 and img_h > 0:
                                area_ratio, aspect, near_full = _box_metrics(d.get('xyxy'), img_w, img_h)
                                if area_ratio <= 0:
                                    continue
                                # keep tiny boxes with class-aware thresholds to avoid undercount
                                if cls_name == 'flower':
                                    if area_ratio < 0.00010 and confv < 0.30:
                                        continue
                                elif cls_name == 'fruit' or cls_name in SPECIFIC_FRUIT_CLASSES:
                                    if area_ratio < 0.0006 and confv < 0.30:
                                        continue
                                elif cls_name in {'tree', 'dumbbell'}:
                                    if area_ratio < 0.0006 and confv < 0.38:
                                        continue
                                elif area_ratio < 0.0006 and confv < 0.45:
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

                        # If custom flower exists, optionally suppress non-custom boxes overlapping custom boxes.
                        # Default behavior keeps COCO results (no suppression).
                        custom_classes = get_custom_class_names()
                        has_custom_flower = any(
                            str(d.get('class', '')).lower().strip() == 'flower'
                            for d in (raw_custom.get('detections', []) if isinstance(raw_custom, dict) else [])
                        )
                        if has_custom_flower and not KEEP_COCO_OVERLAPS_WITH_CUSTOM:
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

                        # Drop COCO fruit labels when custom defines any fruit class.
                        cleaned = suppress_coco_fruit_when_custom_present(cleaned)

                        # Apply source-aware cross-class suppression for sensitive pairs
                        cleaned = resolve_cross_class_overlaps_with_priority(
                            cleaned,
                            iou_thresh=0.60,
                            conf_gap=0.06,
                        )

                        # If custom dumbbell exists, suppress overlapping COCO clock/parking meter boxes.
                        if img_w > 0 and img_h > 0:
                            dumbbells = [
                                d for d in cleaned
                                if (normalize_class_name(d.get('class', '')) or '') == 'dumbbell'
                            ]
                            if dumbbells:
                                reduced = []
                                for d in cleaned:
                                    cls_name = normalize_class_name(d.get('class', '')) or ''
                                    if cls_name in {'parking meter', 'clock'} and str(d.get('_source_model', '')) == 'coco':
                                        overlap_db = any(
                                            _box_iou(d.get('xyxy'), db.get('xyxy')) >= 0.15
                                            and float(db.get('conf', 0.0)) >= 0.20
                                            for db in dumbbells
                                        )
                                        if overlap_db:
                                            continue
                                    reduced.append(d)
                                cleaned = reduced

                        # Remove duplicate boxes of the same class from different models (custom vs COCO).
                        cleaned = suppress_cross_model_same_class_overlaps(
                            cleaned,
                            iou_thresh=0.50,
                            conf_gap=0.06,
                        )

                        merged_dets = dedup_detections_by_class_nms_classwise(cleaned, default_iou=0.45)
                        # Prefer specific fruit labels (apple/banana/orange/...) over generic fruit in hybrid.
                        merged_dets = suppress_generic_fruit_overlaps(
                            merged_dets,
                            iou_thresh=0.25,
                            min_specific_conf=0.40,
                            conf_gap=0.06,
                        )
                        hybrid_stage_stats['after_clean_dedup'] = summarize_detection_stage(merged_dets)

                        # Specialist verification for custom classes to reduce false positives
                        if is_image and raw_img is not None:
                            custom_raw_dets = (raw_custom.get('detections', []) if isinstance(raw_custom, dict) else [])
                            refined_custom = refine_custom_detections_with_specialists(
                                raw_img,
                                custom_raw_dets,
                                base_conf=float(max(0.15, custom_base_conf)),
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
                            hybrid_stage_stats['after_specialist_refine'] = summarize_detection_stage(merged_dets)

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
                            hybrid_stage_stats['after_soft_fallback'] = summarize_detection_stage(merged_dets)

                        # Final verified filter for all classes to reduce over-counting (especially COCO noise)
                        pre_verified_merged = dedup_detections_by_class_nms_classwise(merged_dets, default_iou=0.50)
                        custom_flower_n = int(
                            sum(
                                1
                                for d in (raw_custom.get('detections', []) if isinstance(raw_custom, dict) else [])
                                if (normalize_class_name(d.get('class', '')) or '') == 'flower'
                            )
                        )
                        # In flower-focused scenes, avoid over-strict COCO suppression.
                        strict_coco_mode = not (is_image and custom_flower_n >= 1)
                        merged_dets = verify_and_reduce_detections(
                            merged_dets,
                            img_w=img_w,
                            img_h=img_h,
                            base_conf=float(max(hybrid_conf_floor, min(coco_base_conf, custom_base_conf))),
                            strict_coco=strict_coco_mode,
                        )
                        hybrid_stage_stats['after_verify_reduce'] = summarize_detection_stage(merged_dets)

                        # Root-fix: keep non-empty pre-verified result if strict verifier becomes over-aggressive.
                        if not merged_dets and pre_verified_merged:
                            merged_dets = pre_verified_merged

                        # Flower rescue in hybrid mode: recover missed flowers from custom model.
                        if is_image and raw_img is not None:
                            merged_dets = recover_flower_instances(
                                model=custom_model,
                                image_np=raw_img,
                                base_filtered=merged_dets,
                                tuned_conf=float(max(0.15, custom_base_conf)),
                                imgsz=int(max(int(imgsz), 768)),
                                use_fp16=(torch is not None and torch.cuda.is_available()),
                            )
                            merged_dets = suppress_scene_level_boxes(merged_dets, img_w=img_w, img_h=img_h)
                            merged_dets = enforce_one_object_one_box(merged_dets, img_w=img_w, img_h=img_h)
                            merged_dets = normalize_dense_flower_boxes(merged_dets, img_w=img_w, img_h=img_h)
                            merged_dets = refine_flower_boxes_with_visual_evidence(raw_img, merged_dets, img_w=img_w, img_h=img_h)
                            merged_dets = trim_sparse_flower_outliers(merged_dets, img_w=img_w, img_h=img_h)
                            merged_dets = trim_sparse_custom_outliers(merged_dets, img_w=img_w, img_h=img_h)
                            merged_dets = collapse_sparse_flower_duplicates(merged_dets, img_w=img_w, img_h=img_h)
                            merged_dets = suppress_flower_cross_class_confusions(merged_dets, img_w=img_w, img_h=img_h, base_conf=float(max(0.15, custom_base_conf)))

                            flower_n_h = int(sum(1 for d in merged_dets if (normalize_class_name(d.get('class', '')) or '') == 'flower'))
                            total_n_h = max(1, len(merged_dets))
                            flower_ratio_h = float(flower_n_h) / float(total_n_h)
                            if USE_SPECIALIST_MODELS and flower_n_h <= 12 and flower_n_h >= 2 and flower_ratio_h >= 0.45:
                                flower_model_h = None
                                if os.path.exists(FLOWER_SPECIALIST_MODEL_PATH):
                                    try:
                                        flower_model_h = load_model(FLOWER_SPECIALIST_MODEL_PATH)
                                    except Exception:
                                        flower_model_h = None
                                if flower_model_h is None:
                                    flower_model_h = custom_model

                                recounted_h = recount_flowers_strict(
                                    flower_model=flower_model_h,
                                    image_np=raw_img,
                                    base_detections=merged_dets,
                                    base_conf=float(max(0.15, custom_base_conf)),
                                    imgsz=int(imgsz),
                                    use_fp16=(torch is not None and torch.cuda.is_available()),
                                )
                                recounted_h = suppress_flower_cross_class_confusions(
                                    recounted_h,
                                    img_w=img_w,
                                    img_h=img_h,
                                    base_conf=float(max(0.15, custom_base_conf)),
                                )
                                recounted_h_n = int(sum(1 for d in recounted_h if (normalize_class_name(d.get('class', '')) or '') == 'flower'))
                                if recounted_h_n >= flower_n_h:
                                    merged_dets = recounted_h

                            merged_dets = _apply_scene_context_rules(
                                merged_dets,
                                img_w=img_w,
                                img_h=img_h,
                                base_conf=float(max(hybrid_conf_floor, custom_base_conf)),
                            )
                            hybrid_stage_stats['after_hybrid_flower_recover'] = summarize_detection_stage(merged_dets)
                            if not custom_prefers_generic_fruit():
                                merged_dets = rescue_fruit_from_coco_when_flower_only(
                                    merged_dets,
                                    raw_coco.get('detections', []) if isinstance(raw_coco, dict) else [],
                                    img_w=img_w,
                                    img_h=img_h,
                                    base_conf=float(max(hybrid_conf_floor, min(coco_base_conf, custom_base_conf))),
                                )
                                hybrid_stage_stats['after_coco_fruit_rescue'] = summarize_detection_stage(merged_dets)

                            # Dumbbell rescue: keep low-conf custom dumbbells if they exist.
                            try:
                                raw_custom_dets = raw_custom.get('detections', []) if isinstance(raw_custom, dict) else []
                                rescue_db = []
                                for d in raw_custom_dets:
                                    cls_name = normalize_class_name(d.get('class', '')) or ''
                                    confv = float(d.get('conf', 0.0))
                                    if cls_name != 'dumbbell' or confv < 0.18:
                                        continue
                                    d2 = dict(d)
                                    d2['class'] = 'dumbbell'
                                    d2['_source_model'] = 'custom_rescue'
                                    rescue_db.append(d2)
                                if rescue_db:
                                    merged_dets = dedup_detections_by_class_nms_classwise(
                                        merged_dets + rescue_db,
                                        default_iou=0.60,
                                    )
                                    hybrid_stage_stats['after_dumbbell_rescue'] = summarize_detection_stage(merged_dets)
                            except Exception:
                                pass

                            # Optional force mode: dumbbell-only from custom with very low threshold.
                            force_db_only = bool(st.session_state.get('force_dumbbell_only', False))
                            if force_db_only and custom_model is not None:
                                try:
                                    force_conf = 0.08
                                    force_iou = min(0.55, float(tuned_nms_iou) + 0.05)
                                    force_max_det = max(200, int(max_det))
                                    force_imgsz = int(max(int(imgsz), 960))
                                    _, _, force_dets = run_detection_on_image(
                                        custom_model,
                                        raw_img,
                                        force_conf,
                                        imgsz=force_imgsz,
                                        use_fp16=(torch is not None and torch.cuda.is_available()),
                                        nms_iou=force_iou,
                                        max_det=force_max_det,
                                    )
                                    force_dets = [
                                        d for d in canonicalize_final_detections(force_dets)
                                        if (normalize_class_name(d.get('class', '')) or '') == 'dumbbell'
                                        and float(d.get('conf', 0.0)) >= 0.10
                                    ]
                                    if force_dets:
                                        merged_dets = dedup_detections_by_class_nms_classwise(
                                            force_dets,
                                            default_iou=0.60,
                                        )
                                        hybrid_stage_stats['force_dumbbell_only'] = summarize_detection_stage(merged_dets)
                                except Exception:
                                    pass

                        # Final strict source-isolation merge for Hybrid mode:
                        # COCO keeps COCO classes, Custom keeps 4 custom classes.
                        # This guarantees expected behavior like person (COCO) + flower (Custom).
                        if HYBRID_STRICT_SOURCE_ISOLATION and is_image and raw_img is not None:
                            try:
                                merged_dets = build_hybrid_isolated_merge(
                                    raw_coco.get('detections', []) if isinstance(raw_coco, dict) else [],
                                    raw_custom.get('detections', []) if isinstance(raw_custom, dict) else [],
                                    img_w=img_w,
                                    img_h=img_h,
                                    base_conf=float(max(0.20, hybrid_conf_floor)),
                                )
                                hybrid_stage_stats['after_strict_isolation_merge'] = summarize_detection_stage(merged_dets)
                            except Exception:
                                pass

                        if is_image and raw_img is not None:
                            ann = overlay_detections(raw_img, merged_dets, conf_thresh=float(max(0.10, hybrid_conf_floor)))
                        else:
                            ann = ann_coco if ann_coco is not None else ann_custom

                        # FINAL FILTER: Ensure all detections have _source_model preserved
                        for d in merged_dets:
                            if '_source_model' not in d or not d.get('_source_model'):
                                d['_source_model'] = 'hybrid'
                        
                        final_detections = canonicalize_final_detections(merged_dets)
                        counts = build_counts_from_detections(final_detections)
                        hybrid_stage_stats['final'] = summarize_detection_stage(final_detections)
                        raw = {
                            "detections": final_detections,
                            "mode": "hybrid",
                            "auto_params": hybrid_auto_params,
                            "stage_stats": hybrid_stage_stats,
                            "force_dumbbell_only": bool(st.session_state.get('force_dumbbell_only', False)),
                        }
                    else:
                        ann_coco, counts_coco, raw_coco, raw_img = _run_coco()
                        st.warning(f"Không tìm thấy {CUSTOM_MODEL_PATH}, đang fallback sang COCO.")
                        ann, counts, raw = ann_coco, counts_coco, raw_coco
                else:
                    selected_model = model_path_map.get(model_type, "yolo11n.pt")
                    if selected_model == CUSTOM_MODEL_PATH and os.path.exists(CUSTOM_MODEL_PATH):
                        model = load_custom_model()
                    elif selected_model == COCO_MODEL_PATH:
                        model = load_model(COCO_MODEL_PATH)
                    else:
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
                        max_det=int(st.session_state.get('max_det', 100)),
                        prefer_recall_custom=(selected_model == CUSTOM_MODEL_PATH),
                        mode="custom" if (selected_model == CUSTOM_MODEL_PATH) else "coco",
                    )
                    if is_image and model_type == CUSTOM_ONLY_MODEL_TYPE and raw_img is not None and USE_SPECIALIST_MODELS:
                        refined_custom = refine_custom_detections_with_specialists(
                            raw_img,
                            (raw.get('detections', []) if isinstance(raw, dict) else []),
                            base_conf=float(max(0.15, float(confidence))),
                            imgsz=int(imgsz),
                            use_fp16=(torch is not None and torch.cuda.is_available()),
                        )
                        # FILTER#2: Apply mode filter after specialist refinement
                        refined_custom = filter_detections_by_mode(refined_custom, "custom")
                        
                        ih, iw = raw_img.shape[:2]
                        refined_custom = refine_flower_boxes_with_visual_evidence(raw_img, refined_custom, img_w=iw, img_h=ih)
                        ann = overlay_detections(raw_img, refined_custom, conf_thresh=float(max(0.10, float(confidence))))
                        counts = build_counts_from_detections(refined_custom)
                        raw = {"detections": canonicalize_final_detections(refined_custom), "mode": "custom_refined"}
            except Exception as e:
                st.error(f"Lỗi xử lý: {e}")
                ann, counts, raw = None, {}, {"detections": []}

        latency_ms = int((datetime.now() - t0).total_seconds() * 1000)
        final_detections = extract_final_detections_from_raw(raw)
        uncertain_reason = None

        # Filename-context guard: fruit-labeled images should not keep stray tree boxes.
        try:
            if is_image:
                final_detections = apply_filename_context_guard(final_detections, source_name)
        except Exception:
            pass

        # Hard mode isolation guard (final stage) to prevent class/source leakage.
        try:
            if model_type == CUSTOM_ONLY_MODEL_TYPE:
                final_mode = "custom"
            elif model_type == HYBRID_MODEL_TYPE:
                final_mode = "hybrid"
            else:
                final_mode = "coco"
            final_detections = filter_detections_by_mode(final_detections, final_mode)
        except Exception:
            pass
        # Single source of truth: always rebuild counts from final_detections
        rebuilt_counts = build_counts_from_detections(final_detections)
        counts = rebuilt_counts if rebuilt_counts else (counts if isinstance(counts, dict) else {})

        # If we only had frames before, persist canonical detections for summary/history
        if isinstance(raw, dict):
            raw['detections'] = final_detections

        total_objects = len(final_detections)
        pre_image_safety_detections = canonicalize_final_detections(final_detections)

        # Last safety pass: enforce one-object-one-box before final count/render.
        if is_image:
            try:
                if raw_img is not None and isinstance(raw_img, np.ndarray):
                    ih, iw = raw_img.shape[:2]
                    if model_type == CUSTOM_ONLY_MODEL_TYPE:
                        final_detections = filter_detections_by_mode(final_detections, "custom")
                        if isinstance(raw, dict):
                            raw['detections'] = canonicalize_final_detections(final_detections)
                    else:
                        final_detections = refine_flower_boxes_with_visual_evidence(raw_img, final_detections, img_w=iw, img_h=ih)
                        # Always use one strict finalize path for image count/render.
                        force_db_only = isinstance(raw, dict) and bool(raw.get('force_dumbbell_only'))
                        min_conf_final = 0.10 if force_db_only else float(max(0.10, float(confidence)))
                        finalize_ret = finalize_frame_detections_for_count(
                            final_detections,
                            img_w=iw,
                            img_h=ih,
                            min_conf=float(min_conf_final),
                            debug=True,
                        )
                        if isinstance(finalize_ret, tuple):
                            final_detections = finalize_ret[0]
                            finalize_stage_stats = finalize_ret[1] if len(finalize_ret) > 1 and isinstance(finalize_ret[1], dict) else {}
                            if isinstance(raw, dict):
                                raw['finalize_stage_stats'] = finalize_stage_stats
                        else:
                            final_detections = finalize_ret
                        # Re-apply mode isolation after finalizer (defensive).
                        if model_type == HYBRID_MODEL_TYPE:
                            final_detections = filter_detections_by_mode(final_detections, "hybrid")
                        else:
                            final_detections = filter_detections_by_mode(final_detections, "coco")
                    # Keep render/count consistent with final detections.
                    counts = build_counts_from_detections(final_detections)
                    total_objects = len(final_detections)
                    avg_conf = float(sum(float(d.get('conf', 0.0)) for d in final_detections) / len(final_detections)) if final_detections else 0.0
                    if isinstance(raw, dict):
                        raw['detections'] = canonicalize_final_detections(final_detections)
                    if ann is not None:
                        custom_present = any(
                            (normalize_class_name(d.get('class', '')) or '') in {'flower', 'fruit', 'tree', 'dumbbell'}
                            for d in final_detections
                        )
                        if custom_present:
                            conf_thresh_final = float(max(0.20, float(confidence)))
                        else:
                            conf_thresh_final = float(max(0.10, float(confidence)))
                        ann = overlay_detections(raw_img, final_detections, conf_thresh=conf_thresh_final)
            except Exception:
                pass

        # Custom4 confidence sanity gate:
        # only warn when no confident detections remain; do not forcibly clear valid detections.
        if model_type == CUSTOM_ONLY_MODEL_TYPE and is_image and raw_img is not None:
            try:
                class_floor = {'flower': 0.18, 'fruit': 0.24, 'tree': 0.28, 'dumbbell': 0.36}
                strong = []
                for d in canonicalize_final_detections(final_detections):
                    cls_name = normalize_class_name(d.get('class', '')) or 'unknown'
                    confv = float(d.get('conf', 0.0))
                    floor = float(class_floor.get(cls_name, 0.50))
                    if confv >= floor:
                        strong.append(d)

                if strong:
                    final_detections = canonicalize_final_detections(strong)
                    counts = build_counts_from_detections(final_detections)
                    total_objects = len(final_detections)
                    if isinstance(raw, dict):
                        raw['detections'] = final_detections
                    uncertain_reason = None
                else:
                    final_detections = []
                    counts = {}
                    total_objects = 0
                    if isinstance(raw, dict):
                        raw['detections'] = []
                    uncertain_reason = "Ảnh nằm ngoài phạm vi hiểu biết hiện tại của mô hình Custom4."
            except Exception:
                pass

        if is_image and raw_img is not None and isinstance(raw_img, np.ndarray):
            try:
                custom_present = any(
                    (normalize_class_name(d.get('class', '')) or '') in {'flower', 'fruit', 'tree', 'dumbbell'}
                    for d in final_detections
                )
                if custom_present:
                    conf_thresh_final = float(max(0.20, float(confidence)))
                else:
                    conf_thresh_final = float(max(0.10, float(confidence)))
                ann = overlay_detections(raw_img, final_detections, conf_thresh=conf_thresh_final)
            except Exception:
                pass

        avg_conf = float(sum(float(d.get('conf', 0.0)) for d in final_detections) / len(final_detections)) if final_detections else 0.0
        st.session_state.last_total_objects = total_objects
        st.session_state.last_accuracy = avg_conf * 100.0
        st.session_state.last_latency_ms = latency_ms
        st.session_state.last_load_pct = min(99.0, max(1.0, total_objects * 2.5))
        try:
            if counts:
                top_class = max(counts.items(), key=lambda x: x[1])[0]
                st.session_state.last_top_label = translate_label(str(top_class))
            elif final_detections:
                cls_from_det = str(final_detections[0].get('class', '')).strip()
                st.session_state.last_top_label = translate_label(cls_from_det) if cls_from_det else "Unknown"
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

        done_label = source_display_name or os.path.basename(source_name) or source_name
        st.success(f"Đã phân tích xong tệp: {done_label}")
        if uncertain_reason:
            st.warning(uncertain_reason)

        if SHOW_DETAIL_PANEL:
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
                raw_auto_params = (raw.get('auto_params', {}) if isinstance(raw, dict) else {})
                raw_stage_stats = (raw.get('stage_stats', {}) if isinstance(raw, dict) else {})
                st.json({
                    "tong_vat_the": total_objects,
                    "confidence_tb": round(avg_conf, 3),
                    "thoi_gian_xu_ly_ms": latency_ms,
                    "top_lop": [{"lop": translate_label(k), "so_luong": int(v)} for k, v in top_items],
                    "auto_params": raw_auto_params,
                    "stage_stats": raw_stage_stats,
                    "build": APP_BUILD_ID,
                })
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='sf-card'>", unsafe_allow_html=True)
            if ann is not None:
                st.image(ann, channels="RGB", caption="Kết quả nhận diện thực tế", use_container_width=True)
            else:
                st.info("Chưa có ảnh kết quả để hiển thị.")
            st.markdown("</div>", unsafe_allow_html=True)

        save_analysis(done_label, {"counts": counts, "raw": raw}, media_type=('image' if is_image else 'video'))


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
    # footer removed per request


if __name__ == '__main__':
    main()

