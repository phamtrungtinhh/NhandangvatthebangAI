from pathlib import Path
import os
import shutil

def ensure_model():
    PROJECT_ROOT = Path(__file__).resolve().parent
    target = PROJECT_ROOT / "yolo11n.pt"
    if target.exists():
        return

    repo_id = os.environ.get("HF_MODEL_ID") or os.environ.get("HF_MODEL_REPO")
    filename = os.environ.get("HF_MODEL_FILENAME") or "yolo11n.pt"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if not repo_id:
        print("[startup] HF_MODEL_ID not set and model not found; skipping auto-download.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        print("[startup] huggingface_hub not installed:", e)
        return

    try:
        print(f"[startup] Downloading '{filename}' from '{repo_id}' to {target} ...")
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, use_auth_token=token)
        shutil.copy(downloaded, str(target))
        print("[startup] Model downloaded successfully.")
    except Exception as e:
        print("[startup] Failed to download model:", e)


ensure_model()
