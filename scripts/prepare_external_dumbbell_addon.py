#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(images_dir: Path):
    if not images_dir.exists():
        return
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def filter_label_to_dumbbell(src_label: Path, dst_label: Path) -> int:
    kept = []
    for ln in src_label.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
        except Exception:
            continue
        if cid != 0:
            continue
        # keep normalized box as-is but force class id 0
        kept.append("0 " + " ".join(parts[1:5]))
    if not kept:
        return 0
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(kept), encoding="ascii")
    return len(kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="Source dataset roots, e.g. S:/dumbbell2 S:/dumbbell3")
    ap.add_argument("--out-root", default="S:/workspace/model_custom/dataset/external_dumbbell_addon")
    ap.add_argument("--splits", nargs="+", default=["train", "valid"], help="Source splits to import")
    ap.add_argument("--base-train-list", default="S:/workspace/model_custom/dataset/oversample_db_all_train_x8_hardneg.txt")
    ap.add_argument("--out-train-list", default="S:/workspace/model_custom/dataset/oversample_db_all_train_x8_hardneg_plus_extdb.txt")
    ap.add_argument("--repeat-new", type=int, default=2)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_img = out_root / "train" / "images"
    out_lbl = out_root / "train" / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    imported_imgs = []
    total_boxes = 0

    for src_root_s in args.src:
        src_root = Path(src_root_s)
        tag = src_root.name
        for split in args.splits:
            img_dir = src_root / split / "images"
            lbl_dir = src_root / split / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            for img_path in iter_images(img_dir):
                src_label = lbl_dir / f"{img_path.stem}.txt"
                if not src_label.exists():
                    continue
                new_stem = f"{tag}__{split}__{img_path.stem}"
                dst_img = out_img / f"{new_stem}{img_path.suffix.lower()}"
                dst_lbl = out_lbl / f"{new_stem}.txt"

                kept_n = filter_label_to_dumbbell(src_label, dst_lbl)
                if kept_n <= 0:
                    if dst_lbl.exists():
                        dst_lbl.unlink(missing_ok=True)
                    continue

                shutil.copy2(img_path, dst_img)
                imported_imgs.append(str(dst_img).replace("\\", "/"))
                total_boxes += kept_n

    base_list = [
        x.strip()
        for x in Path(args.base_train_list).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    extra = []
    rep = max(1, int(args.repeat_new))
    for p in imported_imgs:
        extra.extend([p] * rep)

    merged = base_list + extra
    Path(args.out_train_list).write_text("\n".join(merged), encoding="utf-8")

    print("imported_images", len(imported_imgs))
    print("imported_boxes", total_boxes)
    print("repeat_new", rep)
    print("base_train", len(base_list))
    print("merged_train", len(merged))
    print("out_train_list", args.out_train_list)
    print("out_root", str(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
