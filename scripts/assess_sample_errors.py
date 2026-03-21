#!/usr/bin/env python3
import json
from pathlib import Path

IN_JSON = Path('S:/workspace/model_custom/weights/sample_inference_20260312.json')
LABEL_DIR = Path('S:/workspace/model_custom/dataset_flower_focused/val/labels')
OUT_JSON = Path('S:/workspace/model_custom/weights/sample_inference_errors_20260312.json')
FLOWER_CLASS = 1


def parse_labels(path: Path):
    if not path.exists():
        return 0
    cnt = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) >= 1:
            try:
                if int(float(parts[0])) == FLOWER_CLASS:
                    cnt += 1
            except Exception:
                continue
    return cnt


def main():
    data = json.loads(IN_JSON.read_text(encoding='utf-8'))
    results = []
    total = len(data)
    incorrect = []
    for item in data:
        img_path = Path(item['image'])
        stem = img_path.stem
        label_path = LABEL_DIR / (stem + '.txt')
        gt = parse_labels(label_path)
        pred = sum(1 for d in item.get('detections', []) if int(d.get('class_id', -1)) == FLOWER_CLASS)
        ok = (pred == gt)
        results.append({'stem': stem, 'image': str(img_path), 'gt_flower': gt, 'pred_flower': pred, 'match': ok})
        if not ok:
            incorrect.append({'stem': stem, 'gt': gt, 'pred': pred})

    out = {'total_images': total, 'incorrect_count': len(incorrect), 'incorrect': incorrect, 'details': results}
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('Wrote', OUT_JSON)
    print('Total:', total, 'Incorrect:', len(incorrect))


if __name__ == '__main__':
    main()
