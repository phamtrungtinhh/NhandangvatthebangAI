#!/usr/bin/env python3
import json
import random
import subprocess
from pathlib import Path

ROOT = Path('S:/workspace/model_custom/dataset')
IN_DIR = Path('S:/workspace/tmp_test_inputs/selected')
OUT_DIR = Path('S:/workspace/tmp_test_outputs')
INFER_SCRIPT = Path('S:/workspace/scripts/infer_tmp_test.py')

# collect candidate images with GT class 3
candidates = []
for split in ('train','val'):
    lbl_dir = ROOT / split / 'labels'
    img_dir = ROOT / split / 'images'
    if not lbl_dir.exists():
        continue
    for lbl in sorted(lbl_dir.glob('*.txt')):
        txt = lbl.read_text(encoding='utf-8')
        if '\n' not in txt and txt.strip()=='':
            continue
        if any(line.strip().split()[0] == '3' for line in txt.splitlines() if line.strip()):
            # find image
            stem = lbl.stem
            found = None
            for ext in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'):
                p = img_dir / (stem + ext)
                if p.exists():
                    found = p
                    break
            if not found:
                for p in img_dir.glob(stem + '.*'):
                    if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'):
                        found = p
                        break
            if found:
                candidates.append({'image': found, 'label': lbl, 'split': split})

if not candidates:
    print('No candidates found')
    raise SystemExit(1)

sample_n = min(30, len(candidates))
sample = random.sample(candidates, sample_n)
print(f'Running infer for {len(sample)} images')

results = []
for it in sample:
    img = it['image']
    cmd = ['S:/workspace/.venv/Scripts/python.exe', str(INFER_SCRIPT), str(img)]
    print('->', img)
    subprocess.run(cmd, check=True)
    # read infer_log.json
    logp = OUT_DIR / 'infer_log.json'
    if not logp.exists():
        print('no log for', img)
        continue
    out = json.loads(logp.read_text(encoding='utf-8'))
    # find last item matching image
    items = out.get('items', [])
    matched = None
    for item in items:
        if item.get('image') and Path(item.get('image')).name == img.name:
            matched = item
            break
    results.append({'image': str(img), 'report': matched})

OUT = OUT_DIR / 'tree_sample_batch.json'
OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')
print('Wrote', OUT)
