#!/usr/bin/env python3
from pathlib import Path
import random
import json
import os

ROOT = Path("S:/workspace/model_custom/dataset")
OUT_DIR = Path("S:/workspace/tmp_test_outputs/flower_rule_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = {0: "dumbbell", 3: "tree"}
SEED = int(os.environ.get("FLOWER_HARDSET_SEED", "20260317"))
TARGET_N = int(os.environ.get("FLOWER_HARDSET_N", "80"))

def parse_label_area(label_path: Path, images_dir: Path):
    lines = label_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    items = []
    for ln in lines:
        p = ln.strip().split()
        if len(p) < 5:
            continue
        try:
            cid = int(float(p[0]))
            x = float(p[1]); y = float(p[2]); w = float(p[3]); h = float(p[4])
        except Exception:
            continue
        if cid in TARGET_CLASSES:
            # area in normalized coords
            area = max(1e-12, w * h)
            items.append((cid, area))
    return items

def find_image(labels_dir: Path, stem: str):
    for ext in ['.jpg','.jpeg','.png']:
        p = labels_dir.parent / 'images' / (stem + ext)
        if p.exists():
            return p
    return None

def collect_candidates():
    out = []
    for split in ['train','val']:
        labels_dir = ROOT / split / 'labels'
        if not labels_dir.exists():
            continue
        for lbl in labels_dir.glob('*.txt'):
            arr = parse_label_area(lbl, labels_dir)
            if not arr:
                continue
            # difficulty score: smaller min area -> harder
            min_area = min(a for _,a in arr)
            img = find_image(labels_dir, lbl.stem)
            if img is None:
                continue
            out.append({'image': str(img), 'split': split, 'min_area': min_area, 'counts': {TARGET_CLASSES[c]: sum(1 for cid,a in arr if cid==c) for c in set(tc for tc,_ in arr)}})
    return out

def select_hardset(candidates):
    random.seed(SEED)
    # sort by min_area ascending (small boxes first), then shuffle within small buckets
    candidates.sort(key=lambda x: x['min_area'])
    # take top 2/3 smallest by area and sample remainder from rest to increase diversity
    n = min(TARGET_N, len(candidates))
    cutoff = max(1, int(0.66 * len(candidates)))
    smalls = candidates[:cutoff]
    rest = candidates[cutoff:]
    selected = []
    take_smalls = min(n - max(0, n - len(rest)), len(smalls))
    random.shuffle(smalls)
    selected.extend(smalls[:take_smalls])
    remaining = n - len(selected)
    if remaining > 0:
        random.shuffle(rest)
        selected.extend(rest[:remaining])
    # if still short, fill from smalls
    if len(selected) < n:
        selected.extend(smalls[take_smalls:take_smalls + (n - len(selected))])
    return selected[:n]

def main():
    c = collect_candidates()
    sel = select_hardset(c)
    out_json = OUT_DIR / f"hardset_dumbbell_tree_seed{SEED}.json"
    out_list = OUT_DIR / f"hardset_dumbbell_tree_seed{SEED}.txt"
    out_json.write_text(json.dumps({'seed': SEED, 'n': len(sel), 'items': sel}, indent=2), encoding='utf-8')
    out_list.write_text('\n'.join([s['image'] for s in sel]), encoding='utf-8')
    print('wrote', out_json)

if __name__ == '__main__':
    main()
