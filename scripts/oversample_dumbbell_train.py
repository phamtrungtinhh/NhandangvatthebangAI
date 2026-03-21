#!/usr/bin/env python3
import os
from pathlib import Path
import argparse

ROOT = Path("S:/workspace/model_custom/dataset")

def find_label_for_image(img_path: Path, dataset_root: Path) -> Path:
    # img_path is absolute or relative path; find corresponding label in dataset_merged
    p = Path(img_path)
    stem = p.stem
    # search train labels and val labels
    for split in ['train','val']:
        lbl = dataset_root / split / 'labels' / f"{stem}.txt"
        if lbl.exists():
            return lbl
    return None

def has_dumbbell(label_path: Path) -> bool:
    try:
        for ln in label_path.read_text(encoding='utf-8', errors='ignore').splitlines():
            p = ln.strip().split()
            if not p:
                continue
            try:
                cid = int(float(p[0]))
            except Exception:
                continue
            if cid == 0:
                return True
    except Exception:
        return False
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-list', default='S:/workspace/model_custom/dataset/focused03_train.txt')
    parser.add_argument('--out', default='S:/workspace/model_custom/dataset/focused03_train_oversampled.txt')
    parser.add_argument('--mult', type=int, default=3)
    parser.add_argument('--dataset-root', default='S:/workspace/_presentation_archive_20260311/dataset_merged')
    args = parser.parse_args()
    train = Path(args.train_list)
    out = Path(args.out)
    dataset_root = Path(args.dataset_root)
    lines = [l.strip() for l in train.read_text(encoding='utf-8').splitlines() if l.strip()]
    out_lines = []
    for img in lines:
        lbl = find_label_for_image(Path(img), dataset_root)
        if lbl and has_dumbbell(lbl):
            out_lines.extend([img] * args.mult)
        else:
            out_lines.append(img)
    # deduplicate while preserving duplicates inserted (we will not uniquify)
    out.write_text('\n'.join(out_lines), encoding='ascii')
    print('wrote', out)

if __name__ == '__main__':
    main()
