#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(description='Normalize flower labels and export a cleaned dataset copy.')
    parser.add_argument('--input-root', default='S:/workspace/model_custom/dataset', help='Canonical source dataset root.')
    parser.add_argument('--output-root', default='S:/workspace/model_custom/dataset_flower_normalized', help='Output dataset root.')
    parser.add_argument('--report', default='S:/workspace/model_custom/dataset_flower_normalized/normalize_report.json', help='Normalization report path.')
    parser.add_argument('--flower-class-id', type=int, default=1, help='Canonical class id for flower.')
    parser.add_argument('--flower-alias-classes', nargs='*', type=int, default=[], help='Class ids to remap into flower class.')
    parser.add_argument('--tiny-threshold', type=float, default=0.01, help='Drop boxes with width or height below this threshold.')
    parser.add_argument('--dedupe-iou', type=float, default=0.95, help='Remove same-class duplicates above this IoU threshold.')
    parser.add_argument('--dense-mode', choices=['keep', 'split'], default='keep', help='How to handle large dense flower boxes.')
    parser.add_argument('--dense-split-area-threshold', type=float, default=0.08, help='Split flower boxes larger than this normalized area when dense-mode=split.')
    parser.add_argument('--dense-split-grid', type=int, default=2, help='Grid size used to split large flower cluster boxes.')
    parser.add_argument('--overwrite', action='store_true', help='Delete the output dataset root before exporting.')
    return parser.parse_args()


def image_for_label(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        candidate = images_dir / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    return None


def parse_label_file(path: Path):
    boxes = []
    if not path.exists():
        return boxes
    for line_number, raw_line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            continue
        boxes.append({
            'class_id': class_id,
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height,
            'line_number': line_number,
        })
    return boxes


def clip_box(box):
    width = min(max(box['width'], 0.0), 1.0)
    height = min(max(box['height'], 0.0), 1.0)
    x_center = min(max(box['x_center'], 0.0), 1.0)
    y_center = min(max(box['y_center'], 0.0), 1.0)
    return {
        'class_id': int(box['class_id']),
        'x_center': x_center,
        'y_center': y_center,
        'width': width,
        'height': height,
    }


def box_to_corners(box):
    half_w = box['width'] / 2.0
    half_h = box['height'] / 2.0
    return (
        box['x_center'] - half_w,
        box['y_center'] - half_h,
        box['x_center'] + half_w,
        box['y_center'] + half_h,
    )


def intersection_over_union(first, second):
    ax1, ay1, ax2, ay2 = box_to_corners(first)
    bx1, by1, bx2, by2 = box_to_corners(second)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def split_dense_box(box, grid_size):
    child_boxes = []
    cell_width = box['width'] / grid_size
    cell_height = box['height'] / grid_size
    x_start = box['x_center'] - box['width'] / 2.0
    y_start = box['y_center'] - box['height'] / 2.0
    for row in range(grid_size):
        for col in range(grid_size):
            child_boxes.append({
                'class_id': box['class_id'],
                'x_center': x_start + (col + 0.5) * cell_width,
                'y_center': y_start + (row + 0.5) * cell_height,
                'width': cell_width,
                'height': cell_height,
            })
    return child_boxes


def dedupe_boxes(boxes, iou_threshold):
    kept = []
    removed = 0
    for box in sorted(boxes, key=lambda item: (item['class_id'], -item['width'] * item['height'])):
        duplicate = False
        for existing in kept:
            if existing['class_id'] != box['class_id']:
                continue
            if intersection_over_union(existing, box) >= iou_threshold:
                duplicate = True
                removed += 1
                break
        if not duplicate:
            kept.append(box)
    return kept, removed


def normalize_boxes(boxes, args, counters):
    normalized = []
    alias_ids = set(args.flower_alias_classes)
    for box in boxes:
        class_id = int(box['class_id'])
        if class_id in alias_ids:
            class_id = args.flower_class_id
            counters['flower_alias_remapped'] += 1
        clipped = clip_box({**box, 'class_id': class_id})
        if clipped['width'] < args.tiny_threshold or clipped['height'] < args.tiny_threshold:
            counters['tiny_removed'] += 1
            continue
        if clipped['width'] <= 0.0 or clipped['height'] <= 0.0:
            counters['invalid_removed'] += 1
            continue
        area = clipped['width'] * clipped['height']
        if (
            args.dense_mode == 'split'
            and clipped['class_id'] == args.flower_class_id
            and area >= args.dense_split_area_threshold
        ):
            normalized.extend(split_dense_box(clipped, max(2, args.dense_split_grid)))
            counters['dense_cluster_boxes_split'] += 1
            continue
        normalized.append(clipped)
    normalized, duplicates = dedupe_boxes(normalized, args.dedupe_iou)
    counters['duplicate_removed'] += duplicates
    return normalized


def write_labels(path: Path, boxes):
    lines = []
    for box in boxes:
        lines.append(
            f"{int(box['class_id'])} {box['x_center']:.6f} {box['y_center']:.6f} {box['width']:.6f} {box['height']:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def copy_data_yaml(src_root: Path, dst_root: Path):
    src_data = src_root / 'data.yaml'
    dst_data = dst_root / 'data.yaml'
    if not src_data.exists():
        return None
    payload = yaml.safe_load(src_data.read_text(encoding='utf-8')) or {}
    payload['path'] = str(dst_root)
    if 'train' not in payload:
        payload['train'] = 'train/images'
    if 'val' not in payload:
        payload['val'] = 'val/images'
    dst_data.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding='utf-8')
    return dst_data


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    split_stats = defaultdict(lambda: Counter())
    suspicious = []

    for split in ('train', 'val'):
        labels_dir = input_root / split / 'labels'
        images_dir = input_root / split / 'images'
        out_labels_dir = output_root / split / 'labels'
        out_images_dir = output_root / split / 'images'
        out_labels_dir.mkdir(parents=True, exist_ok=True)
        out_images_dir.mkdir(parents=True, exist_ok=True)

        for label_path in sorted(labels_dir.glob('*.txt')):
            image_path = image_for_label(images_dir, label_path.stem)
            if image_path is None:
                suspicious.append({'file': str(label_path), 'issue': 'missing_image'})
                counters['missing_image'] += 1
                continue

            original_boxes = parse_label_file(label_path)
            normalized_boxes = normalize_boxes(original_boxes, args, counters)
            split_stats[split]['label_files'] += 1
            split_stats[split]['boxes_before'] += len(original_boxes)
            split_stats[split]['boxes_after'] += len(normalized_boxes)
            split_stats[split]['flower_boxes_after'] += sum(1 for box in normalized_boxes if box['class_id'] == args.flower_class_id)

            shutil.copy2(image_path, out_images_dir / image_path.name)
            write_labels(out_labels_dir / label_path.name, normalized_boxes)

            if len(normalized_boxes) == 0 and len(original_boxes) > 0:
                suspicious.append({'file': str(label_path), 'issue': 'all_boxes_removed'})

    data_yaml = copy_data_yaml(input_root, output_root)
    report = {
        'input_root': str(input_root),
        'output_root': str(output_root),
        'flower_class_id': args.flower_class_id,
        'flower_alias_classes': list(args.flower_alias_classes),
        'dense_mode': args.dense_mode,
        'dense_split_area_threshold': args.dense_split_area_threshold,
        'dense_split_grid': args.dense_split_grid,
        'tiny_threshold': args.tiny_threshold,
        'dedupe_iou': args.dedupe_iou,
        'counters': dict(counters),
        'split_stats': {split: dict(stats) for split, stats in split_stats.items()},
        'suspicious': suspicious[:200],
        'data_yaml': str(data_yaml) if data_yaml else None,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('Normalization complete')
    print('Output dataset:', output_root)
    if data_yaml:
        print('Data yaml:', data_yaml)
    print('Tiny boxes removed:', counters.get('tiny_removed', 0))
    print('Duplicate boxes removed:', counters.get('duplicate_removed', 0))
    print('Dense cluster boxes split:', counters.get('dense_cluster_boxes_split', 0))
    print('Report written to:', report_path)


if __name__ == '__main__':
    main()