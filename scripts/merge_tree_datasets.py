#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

TARGET_ROOT = Path("S:/workspace/model_custom/dataset")
SOURCES = [
    Path("S:/tree"),
    Path("S:/tree1"),
    Path("S:/tree2"),
]

# Map source split names into target split names.
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TARGET_CLASS_ID = 3


def ensure_dirs() -> None:
    for split in SPLIT_MAP.values():
        (TARGET_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (TARGET_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def pick_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for p in images_dir.glob(f"{stem}.*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def unique_stem(target_images_dir: Path, target_labels_dir: Path, base_stem: str) -> str:
    candidate = base_stem
    idx = 1
    while True:
        image_conflict = any((target_images_dir / f"{candidate}{ext}").exists() for ext in IMAGE_EXTS)
        label_conflict = (target_labels_dir / f"{candidate}.txt").exists()
        if not image_conflict and not label_conflict:
            return candidate
        candidate = f"{base_stem}__{idx}"
        idx += 1


def remap_label_to_tree(src_label: Path, dst_label: Path) -> int:
    text = src_label.read_text(encoding="utf-8", errors="ignore")
    out_lines: list[str] = []
    obj_count = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            # Skip malformed lines here; dataset audit will report if needed.
            continue
        parts[0] = str(TARGET_CLASS_ID)
        out_lines.append(" ".join(parts[:5]))
        obj_count += 1

    dst_label.write_text(("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8")
    return obj_count


def merge_source(src_root: Path) -> dict:
    stats = {
        "source": str(src_root),
        "copied_pairs": 0,
        "objects_written": 0,
        "skipped_missing_image": 0,
        "skipped_missing_label": 0,
        "renamed_due_conflict": 0,
    }

    src_tag = src_root.name.lower()

    for src_split, dst_split in SPLIT_MAP.items():
        src_images_dir = src_root / src_split / "images"
        src_labels_dir = src_root / src_split / "labels"
        if not src_labels_dir.exists():
            continue

        dst_images_dir = TARGET_ROOT / dst_split / "images"
        dst_labels_dir = TARGET_ROOT / dst_split / "labels"

        for lbl in src_labels_dir.glob("*.txt"):
            stem = lbl.stem
            img = pick_image_for_stem(src_images_dir, stem)
            if img is None:
                stats["skipped_missing_image"] += 1
                continue
            if not lbl.exists():
                stats["skipped_missing_label"] += 1
                continue

            base_stem = f"{src_tag}_{src_split}_{stem}"
            out_stem = unique_stem(dst_images_dir, dst_labels_dir, base_stem)
            if out_stem != base_stem:
                stats["renamed_due_conflict"] += 1

            dst_img = dst_images_dir / f"{out_stem}{img.suffix.lower()}"
            dst_lbl = dst_labels_dir / f"{out_stem}.txt"

            shutil.copy2(img, dst_img)
            stats["objects_written"] += remap_label_to_tree(lbl, dst_lbl)
            stats["copied_pairs"] += 1

    return stats


def main() -> None:
    ensure_dirs()
    all_stats: list[dict] = []
    for src in SOURCES:
        if not src.exists():
            print(f"[WARN] Source not found: {src}")
            continue
        st = merge_source(src)
        all_stats.append(st)

    print("Merge completed")
    total_pairs = 0
    total_objects = 0
    for st in all_stats:
        total_pairs += st["copied_pairs"]
        total_objects += st["objects_written"]
        print(
            f"- {st['source']}: pairs={st['copied_pairs']}, objects={st['objects_written']}, "
            f"renamed={st['renamed_due_conflict']}, missing_image={st['skipped_missing_image']}"
        )
    print(f"TOTAL pairs={total_pairs}, objects={total_objects}")


if __name__ == "__main__":
    main()
