import streamlit as st
import sqlite3
import json
import time
import os
import tempfile
from datetime import datetime

from PIL import Image
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

DB_PATH = "analysis.db"
# default relative path; we'll attempt to auto-detect other common locations if missing
FLOWER_MODEL_PATH = "models/flower_best.pt"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)


def find_flower_model(preferred: str = None):
    """Return an existing model path from a list of common candidates.
    If `preferred` is provided and exists, it will be returned.
    """
    candidates = []

    def _add_path(path_str: str):
        if not path_str:
            return
        candidates.append(path_str)
        if not os.path.isabs(path_str):
            candidates.append(os.path.join(APP_DIR, path_str))
            candidates.append(os.path.join(PROJECT_ROOT, path_str))

    if preferred:
        _add_path(preferred)
    # common local locations
    _add_path("models/flower_best.pt")
    _add_path("models/flower_best.pth")
    # runs output (last training run)
    try:
        import glob
        run_patterns = [
            os.path.join("runs", "detect", "*", "weights", "best.pt"),
            os.path.join(APP_DIR, "runs", "detect", "*", "weights", "best.pt"),
            os.path.join(PROJECT_ROOT, "runs", "detect", "*", "weights", "best.pt"),
        ]
        runs = []
        for pattern in run_patterns:
            runs.extend(glob.glob(pattern))
        runs = sorted(set(runs), key=os.path.getmtime, reverse=True)
        candidates += runs
    except Exception:
        pass

    # absolute/other fallback candidates
    candidates += [
        os.path.join(os.getcwd(), "models", "flower_best.pt"),
        os.path.join(APP_DIR, "models", "flower_best.pt"),
        os.path.join(PROJECT_ROOT, "models", "flower_best.pt"),
    ]

    for c in candidates:
        try:
            if c and os.path.exists(c):
                return c
        except Exception:
            continue
    return None


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp TEXT,
            objects_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


@st.cache_resource
def load_model(model_name: str = FLOWER_MODEL_PATH):
    if YOLO is None:
        raise RuntimeError("`ultralytics` package is not available. Install requirements.")
    # Accept both 'yolov8n' and explicit filenames
    if not model_name.endswith('.pt'):
        model_name = model_name + ".pt"
    model = YOLO(model_name)
    return model


def annotate_image(img: np.ndarray, results, names_map=None, conf_thresh=0.25):
    img = img.copy()
    h, w = img.shape[:2]
    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
            cls = int(box.cls[0]) if hasattr(box, 'cls') else None
            if conf < conf_thresh:
                continue
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else None
            if xyxy is None:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            label = names_map.get(cls, str(cls)) if names_map else str(cls)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def run_detection_on_image(model, image_np, conf: float):
    results = model(image_np, conf=conf)
    names = results[0].names if hasattr(results[0], 'names') else {}
    ann = annotate_image(image_np, results, names, conf_thresh=conf)
    # collect objects
    counts = {}
    for r in results:
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        for box in boxes:
            confv = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
            if confv < conf:
                continue
            cls = int(box.cls[0]) if hasattr(box, 'cls') else None
            name = names.get(cls, str(cls))
            counts[name] = counts.get(name, 0) + 1
    return ann, counts


def run_detection_on_video(model, video_bytes, conf: float, max_frames=30, sample_rate=5):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmp.write(video_bytes)
    tmp.flush()
    tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    frame_idx = 0
    aggregated = {}
    first_annotated = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img, conf=conf)
            names = results[0].names if hasattr(results[0], 'names') else {}
            # collect
            for r in results:
                boxes = getattr(r, 'boxes', None)
                if boxes is None:
                    continue
                for box in boxes:
                    confv = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    if confv < conf:
                        continue
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else None
                    name = names.get(cls, str(cls))
                    aggregated[name] = aggregated.get(name, 0) + 1
            if first_annotated is None:
                first_annotated = annotate_image(img, results, names, conf_thresh=conf)
        frame_idx += 1
        if frame_idx >= max_frames:
            break
    cap.release()
    try:
        os.unlink(tmp.name)
    except Exception:
        pass
    if first_annotated is None:
        # create thumbnail from black
        first_annotated = np.zeros((480, 640, 3), dtype=np.uint8)
    return first_annotated, aggregated


def save_analysis(filename: str, objects: dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO analyses (filename, timestamp, objects_json) VALUES (?, ?, ?)",
                (filename, datetime.utcnow().isoformat(), json.dumps(objects, ensure_ascii=False)))
    conn.commit()
    conn.close()


def list_history():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, filename, timestamp, objects_json FROM analyses ORDER BY id DESC LIMIT 200")
    rows = cur.fetchall()
    conn.close()
    return rows


def main():
    st.set_page_config(page_title="AI Object Detector", layout="wide")
    st.title("Hệ thống nhận diện và phân tích vật thể (Streamlit)")

    init_db()

    with st.sidebar:
        st.header("Cấu hình")
        # attempt to find a model automatically
        auto_model = find_flower_model(FLOWER_MODEL_PATH)
        model_choice = st.text_input("Đường dẫn model hoa", value=auto_model or FLOWER_MODEL_PATH)
        if model_choice and os.path.exists(model_choice):
            st.success(f"Model hiện tại: {model_choice}")
        else:
            st.warning(f"Model không tồn tại: {model_choice}")
        conf = st.slider("Confidence", 0.01, 0.9, 0.25, 0.01)
        show_history = st.checkbox("Hiển thị lịch sử phân tích", value=True)

    st.markdown("Kéo thả hoặc chọn tệp ảnh/video (ngắn). Dữ liệu gốc không được lưu, chỉ lưu meta.")
    uploaded = st.file_uploader("Chọn ảnh hoặc video", type=["png", "jpg", "jpeg", "mp4", "mov"], accept_multiple_files=False)

    model = None
    if uploaded is not None:
        if st.button("Chạy nhận diện"):
            resolved = model_choice
            if not resolved or not os.path.exists(resolved):
                # try auto-detect
                resolved = find_flower_model(model_choice)
            if not resolved or not os.path.exists(resolved):
                st.error(f"Không tìm thấy model hoa tại: {model_choice}")
                st.info("Bạn có thể nhập đường dẫn chính xác trong sidebar hoặc đặt model vào `models/flower_best.pt`")
                return
            try:
                with st.spinner("Tải model và chạy nhận diện..."):
                    model = load_model(resolved)
            except Exception as e:
                st.error(f"Không thể tải model: {e}")
                return

            fname = uploaded.name
            data = uploaded.read()
            is_image = uploaded.type.startswith('image')
            if is_image:
                pil = Image.open(uploaded)
                img = np.array(pil.convert('RGB'))
                ann, counts = run_detection_on_image(model, img, conf)
                st.image(ann, channels="RGB", use_column_width=True)
            else:
                ann, counts = run_detection_on_video(model, data, conf)
                st.image(ann, channels="RGB", use_column_width=True)

            st.subheader("Vật thể phát hiện")
            if counts:
                for k, v in counts.items():
                    youtube_q = f"https://www.youtube.com/results?search_query={k.replace(' ', '+')}"
                    shopee_q = f"https://shopee.vn/search?keyword={k.replace(' ', '%20')}"
                    st.markdown(f"- **{k}**: {v}  [YouTube]({youtube_q}) | [Shopee]({shopee_q})")
            else:
                st.write("Không tìm thấy vật thể ở ngưỡng confidence hiện tại.")

            save_analysis(fname, counts)

    if show_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Lịch sử gần đây")
        rows = list_history()
        for r in rows[:20]:
            id_, fname, ts, objs = r
            objs_parsed = json.loads(objs)
            summary = ", ".join([f"{k}({v})" for k, v in objs_parsed.items()]) if objs_parsed else "(trống)"
            st.sidebar.markdown(f"**{fname}**  {ts}  {summary}")


if __name__ == '__main__':
    main()
