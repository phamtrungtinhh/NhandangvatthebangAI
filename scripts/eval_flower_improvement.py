#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

FLOWER_CLASS_ID = 1
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(description='Compare before/after flower performance on sparse and dense scenes.')
    parser.add_argument('--baseline-model', required=True, help='Baseline model path.')
    parser.add_argument('--candidate-model', required=True, help='Candidate model path.')
    parser.add_argument('--dataset-root', default='S:/workspace/model_custom/dataset_flower_focused', help='Focused dataset root.')
    parser.add_argument('--split', default='val', choices=['train', 'val'], help='Dataset split to evaluate.')
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--dense-threshold', type=int, default=5)
    parser.add_argument('--output', default='S:/workspace/model_custom/weights/flower_eval_improvement.json', help='Evaluation report path.')
    return parser.parse_args()


def find_image(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        candidate = images_dir / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    return None


def parse_label_file(path: Path):
    boxes = []
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        boxes.append({
            'class_id': int(float(parts[0])),
            'x_center': float(parts[1]),
            'y_center': float(parts[2]),
            'width': float(parts[3]),
            'height': float(parts[4]),
        })
    return boxes


def gather_records(dataset_root: Path, split: str):
    images_dir = dataset_root / split / 'images'
    labels_dir = dataset_root / split / 'labels'
    records = []
    for label_path in sorted(labels_dir.glob('*.txt')):
        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            continue
        boxes = parse_label_file(label_path)
        flower_count = sum(1 for box in boxes if box['class_id'] == FLOWER_CLASS_ID)
        if flower_count <= 0:
            continue
        records.append({'stem': label_path.stem, 'image_path': image_path, 'flower_count': flower_count})
    return records


def evaluate_model(model_path: Path, records, conf: float, imgsz: int, device: str, dense_threshold: int):
    model = YOLO(str(model_path))
    grouped = defaultdict(list)
    items = []
    for record in records:
        result = model.predict(source=str(record['image_path']), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        pred_flower = 0
        total_pred = 0
        if getattr(result, 'boxes', None) is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.tolist()
            total_pred = len(classes)
            pred_flower = sum(1 for class_id in classes if int(class_id) == FLOWER_CLASS_ID)

        gt_flower = record['flower_count']
        group = 'dense' if gt_flower >= dense_threshold else 'sparse'
        grouped[group].append({'gt': gt_flower, 'pred': pred_flower, 'total_pred': total_pred, 'stem': record['stem']})
        items.append({'stem': record['stem'], 'group': group, 'gt_flower': gt_flower, 'pred_flower': pred_flower, 'pred_total': total_pred})

    summary = {}
    for group_name, group_items in grouped.items():
        counter = Counter()
        abs_error = 0
        for item in group_items:
            if item['pred'] == 0:
                counter['zero_pred'] += 1
            if item['pred'] < item['gt']:
                counter['undercount'] += 1
            elif item['pred'] > item['gt']:
                counter['overcount'] += 1
            else:
                counter['exact'] += 1
            abs_error += abs(item['pred'] - item['gt'])
        summary[group_name] = {
            'images': len(group_items),
            'avg_gt_flower': sum(item['gt'] for item in group_items) / len(group_items),
            'avg_pred_flower': sum(item['pred'] for item in group_items) / len(group_items),
            'avg_pred_total': sum(item['total_pred'] for item in group_items) / len(group_items),
            'mean_abs_count_error': abs_error / len(group_items),
            'zero_pred': counter['zero_pred'],
            'undercount': counter['undercount'],
            'overcount': counter['overcount'],
            'exact': counter['exact'],
        }
    return {'model': str(model_path), 'summary': summary, 'items': items}


def build_delta(baseline_summary, candidate_summary):
    delta = {}
    for group_name in sorted(set(baseline_summary) | set(candidate_summary)):
        base = baseline_summary.get(group_name, {})
        cand = candidate_summary.get(group_name, {})
        if not base or not cand:
            continue
        delta[group_name] = {
            'mean_abs_count_error_delta': cand['mean_abs_count_error'] - base['mean_abs_count_error'],
            'zero_pred_delta': cand['zero_pred'] - base['zero_pred'],
            'undercount_delta': cand['undercount'] - base['undercount'],
            'overcount_delta': cand['overcount'] - base['overcount'],
            'exact_delta': cand['exact'] - base['exact'],
        }
    return delta


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    records = gather_records(dataset_root, args.split)
    baseline = evaluate_model(Path(args.baseline_model), records, args.conf, args.imgsz, args.device, args.dense_threshold)
    candidate = evaluate_model(Path(args.candidate_model), records, args.conf, args.imgsz, args.device, args.dense_threshold)
    payload = {
        'dataset_root': str(dataset_root),
        'split': args.split,
        'conf': args.conf,
        'imgsz': args.imgsz,
        'dense_threshold': args.dense_threshold,
        'records': len(records),
        'baseline': baseline,
        'candidate': candidate,
        'delta': build_delta(baseline['summary'], candidate['summary']),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print('Flower evaluation complete')
    print('Evaluated records:', len(records))
    for label, summary in (('Baseline', baseline['summary']), ('Candidate', candidate['summary'])):
        print(label)
        for group_name, metrics in summary.items():
            print(
                f"  {group_name}: images={metrics['images']}, mae={metrics['mean_abs_count_error']:.3f}, "
                f"zero={metrics['zero_pred']}, under={metrics['undercount']}, over={metrics['overcount']}, exact={metrics['exact']}"
            )
    print('Report written to:', output_path)


if __name__ == '__main__':
    main()