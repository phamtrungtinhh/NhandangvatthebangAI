#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def find_label_for_image(img_path: Path, dataset_root: Path) -> Path | None:
    # First, try sibling labels path from the image path itself.
    p = Path(img_path)
    cand = Path(str(p).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')).with_suffix('.txt')
    if cand.exists():
        return cand

    stem = p.stem
    for split in ("train", "val"):
        lbl = dataset_root / split / "labels" / f"{stem}.txt"
        if lbl.exists():
            return lbl
    # Extra fallback: some workspaces keep labels under model_custom/dataset.
    alt_root = Path("S:/workspace/model_custom/dataset")
    for split in ("train", "val"):
        lbl = alt_root / split / "labels" / f"{stem}.txt"
        if lbl.exists():
            return lbl
    return None


def parse_class_ids(label_path: Path) -> set[int]:
    out: set[int] = set()
    try:
        for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = ln.strip().split()
            if not p:
                continue
            try:
                out.add(int(float(p[0])))
            except Exception:
                continue
    except Exception:
        return set()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-list", default="S:/workspace/model_custom/dataset/focused03_train.txt")
    ap.add_argument("--dataset-root", default="S:/workspace/_presentation_archive_20260311/dataset_merged")
    ap.add_argument("--db-mult", type=int, default=8)
    ap.add_argument("--hard-neg-repeat", type=int, default=2)
    ap.add_argument("--hard-neg-cap", type=int, default=0, help="0 means use all")
    ap.add_argument("--seed", type=int, default=20260317)
    ap.add_argument("--out-db", default="S:/workspace/model_custom/dataset/focused03_train_oversampled_x8.txt")
    ap.add_argument("--out-hard-neg", default="S:/workspace/model_custom/dataset/focused03_train_hardneg_flowerfruit.txt")
    ap.add_argument("--out-combined", default="S:/workspace/model_custom/dataset/focused03_train_dbx8_hardneg.txt")
    args = ap.parse_args()

    train_list = Path(args.train_list)
    dataset_root = Path(args.dataset_root)
    out_db = Path(args.out_db)
    out_hn = Path(args.out_hard_neg)
    out_combined = Path(args.out_combined)

    lines = [x.strip() for x in train_list.read_text(encoding="utf-8").splitlines() if x.strip()]

    db_imgs: list[str] = []
    hard_negs: list[str] = []
    normal_imgs: list[str] = []

    for img in lines:
        lbl = find_label_for_image(Path(img), dataset_root)
        if lbl is None:
            normal_imgs.append(img)
            continue

        class_ids = parse_class_ids(lbl)
        has_db = 0 in class_ids
        has_flower_or_fruit = (1 in class_ids) or (2 in class_ids)

        if has_db:
            db_imgs.append(img)
        else:
            normal_imgs.append(img)
            if has_flower_or_fruit:
                hard_negs.append(img)

    rng = random.Random(args.seed)
    if args.hard_neg_cap > 0 and len(hard_negs) > args.hard_neg_cap:
        hard_negs = rng.sample(hard_negs, args.hard_neg_cap)

    db_oversampled: list[str] = []
    for img in lines:
        if img in db_imgs:
            db_oversampled.extend([img] * max(1, int(args.db_mult)))
        else:
            db_oversampled.append(img)

    hard_neg_manifest: list[str] = []
    for img in hard_negs:
        hard_neg_manifest.extend([img] * max(1, int(args.hard_neg_repeat)))

    combined = db_oversampled + hard_neg_manifest

    out_db.write_text("\n".join(db_oversampled), encoding="ascii")
    out_hn.write_text("\n".join(hard_neg_manifest), encoding="ascii")
    out_combined.write_text("\n".join(combined), encoding="ascii")

    print("wrote", out_db)
    print("wrote", out_hn)
    print("wrote", out_combined)
    print(
        {
            "total_base": len(lines),
            "db_images": len(db_imgs),
            "hard_neg_images": len(hard_negs),
            "db_mult": int(args.db_mult),
            "hard_neg_repeat": int(args.hard_neg_repeat),
            "total_combined": len(combined),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
