#!/usr/bin/env python3
import json
from pathlib import Path


CLASS_ID_FLOWER = 1
CLASS_NAMES = {0: 'dumbbell', 1: 'flower', 2: 'fruit', 3: 'tree'}
BIAS_JSON = Path('S:/workspace/model_custom/weights/val_bias_check.json')
VAL_LABEL_DIR = Path('S:/workspace/model_custom/dataset/val/labels')


def load_predictions():
    data = json.loads(BIAS_JSON.read_text(encoding='utf-8'))
    by_stem = {}
    for item in data.get('items', []):
        by_stem[Path(item['image']).stem] = item
    return data, by_stem


def parse_label_file(path: Path):
    boxes = []
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        _, _, width, height = map(float, parts[1:])
        boxes.append({
            'class_id': class_id,
            'width': width,
            'height': height,
        })
    return boxes


def summarize_group(name, samples):
    print(f'[{name}]')
    print(f'Images: {len(samples)}')
    if not samples:
        print()
        return

    gt_avg = sum(item['gt_flower'] for item in samples) / len(samples)
    pred_flower_avg = sum(item['pred_flower'] for item in samples) / len(samples)
    pred_total_avg = sum(item['pred_total'] for item in samples) / len(samples)
    width_avg = sum(item['avg_width'] for item in samples) / len(samples)
    height_avg = sum(item['avg_height'] for item in samples) / len(samples)
    zero_pred = sum(1 for item in samples if item['pred_flower'] == 0)
    undercount = sum(1 for item in samples if item['pred_flower'] < item['gt_flower'])
    overcount = sum(1 for item in samples if item['pred_flower'] > item['gt_flower'])
    exact = sum(1 for item in samples if item['pred_flower'] == item['gt_flower'])

    print(f'Average GT flower count: {gt_avg:.2f}')
    print(f'Average predicted flower count: {pred_flower_avg:.2f}')
    print(f'Average predicted total detections: {pred_total_avg:.2f}')
    print(f'Average GT flower bbox size: w={width_avg:.4f}, h={height_avg:.4f}')
    print(f'Zero flower predictions: {zero_pred}/{len(samples)}')
    print(f'Undercount: {undercount}/{len(samples)}')
    print(f'Overcount: {overcount}/{len(samples)}')
    print(f'Exact count: {exact}/{len(samples)}')

    worst = sorted(samples, key=lambda item: (abs(item['pred_flower'] - item['gt_flower']), item['gt_flower']), reverse=True)[:8]
    print('Examples:')
    for item in worst:
        print(
            f"  {item['stem']}: gt_flower={item['gt_flower']}, pred_flower={item['pred_flower']}, "
            f"pred_total={item['pred_total']}"
        )
    print()


def main():
    _, predictions = load_predictions()
    flower_samples = []
    prediction_confusion = {class_id: 0 for class_id in CLASS_NAMES}

    for label_path in sorted(VAL_LABEL_DIR.glob('*.txt')):
        gt_boxes = parse_label_file(label_path)
        gt_flower_boxes = [box for box in gt_boxes if box['class_id'] == CLASS_ID_FLOWER]
        if not gt_flower_boxes:
            continue

        prediction_item = predictions.get(label_path.stem, {'detections': [], 'num_detections': 0})
        pred_flower = sum(1 for det in prediction_item['detections'] if det['class_id'] == CLASS_ID_FLOWER)
        for det in prediction_item['detections']:
            prediction_confusion[int(det['class_id'])] += 1

        flower_samples.append({
            'stem': label_path.stem,
            'gt_flower': len(gt_flower_boxes),
            'pred_flower': pred_flower,
            'pred_total': int(prediction_item.get('num_detections', 0)),
            'avg_width': sum(box['width'] for box in gt_flower_boxes) / len(gt_flower_boxes),
            'avg_height': sum(box['height'] for box in gt_flower_boxes) / len(gt_flower_boxes),
        })

    sparse = [item for item in flower_samples if item['gt_flower'] <= 2]
    medium = [item for item in flower_samples if 3 <= item['gt_flower'] <= 4]
    dense = [item for item in flower_samples if item['gt_flower'] >= 5]

    print('Flower failure-mode analysis')
    print('===========================')
    print(f'Total flower-labeled validation images: {len(flower_samples)}')
    print()
    summarize_group('Sparse scenes (1-2 flowers)', sparse)
    summarize_group('Medium scenes (3-4 flowers)', medium)
    summarize_group('Dense scenes (>=5 flowers)', dense)

    print('[Predicted classes on flower-only GT images]')
    total_confusion = sum(prediction_confusion.values())
    for class_id in sorted(prediction_confusion):
        count = prediction_confusion[class_id]
        pct = (count / total_confusion * 100.0) if total_confusion else 0.0
        print(f"{class_id} ({CLASS_NAMES[class_id]}): {count} ({pct:.2f}%)")


if __name__ == '__main__':
    main()