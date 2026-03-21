#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter
import shutil

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge dumbbell datasets into main dataset with class-safe labels")
    parser.add_argument("--dest", default="S:/workspace/model_custom/dataset", help="Destination YOLO dataset root")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "Source dataset spec: NAME=PATH (e.g. db1=S:/dumbbell). "
            "May be passed multiple times."
        ),
    )
    parser.add_argument("--prefix", default="db", help="Filename prefix for merged files")
    parser.add_argument("--dry-run", action="store_true", help="Do not copy files, only print actions")
    return parser.parse_args()


def parse_source_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --source value: {spec}. Expected NAME=PATH")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        src_path = Path(raw_path.strip())
        if not name:
            raise ValueError(f"Invalid source name in: {spec}")
        parsed.append((name, src_path))
    return parsed


def find_image_by_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    # fallback: case-insensitive extension mismatch
    matches = [p for p in images_dir.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTS]
    if len(matches) == 1:
        return matches[0]
    return None


def next_index(dest_images: Path, prefix: str) -> int:
    max_idx = -1
    needle = f"{prefix}_"
    for p in dest_images.glob(f"{prefix}_*"):
        stem = p.stem
        if not stem.startswith(needle):
            continue
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        idx_s = parts[1]
        if idx_s.isdigit():
            max_idx = max(max_idx, int(idx_s))
    return max_idx + 1


def normalize_dumbbell_lines(raw_lines: list[str]) -> tuple[list[str], Counter]:
    out: list[str] = []
    stat = Counter()
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            stat["bad_format"] += 1
            continue
        try:
            cls_id = int(float(parts[0]))
        except Exception:
            stat["bad_class"] += 1
            continue
        if cls_id != 0:
            stat["dropped_non_dumbbell"] += 1
            continue
        # force class id to 0 and keep bbox values as-is
        out.append(" ".join(["0", parts[1], parts[2], parts[3], parts[4]]))
        stat["kept"] += 1
    return out, stat


def ensure_dirs(root: Path) -> None:
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    dest_root = Path(args.dest)
    ensure_dirs(dest_root)

    sources = parse_source_specs(args.source)

    split_map = {
        "train": "train",
        "valid": "val",
    }

    summary = Counter()
    idx_by_split = {
        "train": next_index(dest_root / "train" / "images", args.prefix),
        "val": next_index(dest_root / "val" / "images", args.prefix),
    }

    for src_name, src_root in sources:
        if not src_root.exists():
            print(f"[WARN] Source not found: {src_root}")
            summary["missing_source"] += 1
            continue

        for src_split, dst_split in split_map.items():
            src_images = src_root / src_split / "images"
            src_labels = src_root / src_split / "labels"
            dst_images = dest_root / dst_split / "images"
            dst_labels = dest_root / dst_split / "labels"

            if not src_images.exists() or not src_labels.exists():
                print(f"[INFO] Skip split {src_name}:{src_split} (missing images/labels)")
                continue

            label_files = sorted(src_labels.glob("*.txt"))
            for lbl in label_files:
                stem = lbl.stem
                img = find_image_by_stem(src_images, stem)
                if img is None:
                    summary["missing_image_for_label"] += 1
                    continue

                raw_lines = lbl.read_text(encoding="utf-8", errors="ignore").splitlines()
                lines, stat = normalize_dumbbell_lines(raw_lines)
                summary.update(stat)

                # only keep samples that still contain dumbbell boxes
                if not lines:
                    summary["skipped_no_dumbbell_box"] += 1
                    continue

                idx = idx_by_split[dst_split]
                idx_by_split[dst_split] += 1
                new_stem = f"{args.prefix}_{idx:06d}_{src_name}"
                new_img = dst_images / f"{new_stem}{img.suffix.lower()}"
                new_lbl = dst_labels / f"{new_stem}.txt"

                # ultra-safe collision check (should not happen with idx sequence)
                while new_img.exists() or new_lbl.exists():
                    idx = idx_by_split[dst_split]
                    idx_by_split[dst_split] += 1
                    new_stem = f"{args.prefix}_{idx:06d}_{src_name}"
                    new_img = dst_images / f"{new_stem}{img.suffix.lower()}"
                    new_lbl = dst_labels / f"{new_stem}.txt"

                if args.dry_run:
                    print(f"[DRY] {img} -> {new_img}")
                    print(f"[DRY] {lbl} -> {new_lbl}")
                else:
                    shutil.copy2(img, new_img)
                    new_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

                summary[f"copied_{dst_split}"] += 1
                summary["copied_total"] += 1

    print("\n=== Merge summary ===")
    for k in sorted(summary):
        print(f"{k}: {summary[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
