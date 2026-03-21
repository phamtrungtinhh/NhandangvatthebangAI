#!/usr/bin/env python3
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
import yaml

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'}
FLOWER_CLASS_ID = 1


def parse_args():
    parser = argparse.ArgumentParser(description='Build a flower-focused 4-class dataset with sparse and dense augmentations.')
    parser.add_argument('--input-root', default='S:/workspace/model_custom/dataset_flower_normalized', help='Normalized dataset root.')
    parser.add_argument('--output-root', default='S:/workspace/model_custom/dataset_flower_focused', help='Focused dataset root.')
    parser.add_argument('--bias-json', default='S:/workspace/model_custom/weights/val_bias_check.json', help='Validation bias json used to mine hard flower cases.')
    parser.add_argument('--manifest', default='S:/workspace/model_custom/dataset_flower_focused/manifest.json', help='Manifest output path.')
    parser.add_argument('--hard-val-count', type=int, default=64, help='Number of hard flower validation images to keep in focused val set.')
    parser.add_argument('--dense-threshold', type=int, default=5, help='Flower count threshold treated as dense.')
    parser.add_argument('--sparse-threshold', type=int, default=2, help='Flower count threshold treated as sparse.')
    parser.add_argument('--tile-size', type=int, default=640, help='Tile size for dense flower crops.')
    parser.add_argument('--tile-overlap', type=float, default=0.2, help='Dense crop overlap ratio.')
    parser.add_argument('--min-box-visibility', type=float, default=0.4, help='Minimum retained intersection ratio for cropped boxes.')
    parser.add_argument('--overwrite', action='store_true', help='Delete the focused dataset root before writing.')
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


def write_label_file(path: Path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{int(box['class_id'])} {box['x_center']:.6f} {box['y_center']:.6f} {box['width']:.6f} {box['height']:.6f}"
        for box in boxes
    ]
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def load_split_records(dataset_root: Path, split: str):
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
        records.append({
            'stem': label_path.stem,
            'split': split,
            'image_path': image_path,
            'label_path': label_path,
            'boxes': boxes,
            'flower_count': flower_count,
        })
    return records


def copy_record(record, output_root: Path, split: str, suffix: str = ''):
    out_image_dir = output_root / split / 'images'
    out_label_dir = output_root / split / 'labels'
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    image_name = record['image_path'].stem + suffix + record['image_path'].suffix.lower()
    label_name = record['label_path'].stem + suffix + '.txt'
    shutil.copy2(record['image_path'], out_image_dir / image_name)
    write_label_file(out_label_dir / label_name, record['boxes'])
    return image_name, label_name


def horizontal_flip_boxes(boxes):
    transformed = []
    for box in boxes:
        transformed.append({
            'class_id': box['class_id'],
            'x_center': 1.0 - box['x_center'],
            'y_center': box['y_center'],
            'width': box['width'],
            'height': box['height'],
        })
    return transformed


def rotate_boxes_90_cw(boxes):
    transformed = []
    for box in boxes:
        transformed.append({
            'class_id': box['class_id'],
            'x_center': 1.0 - box['y_center'],
            'y_center': box['x_center'],
            'width': box['height'],
            'height': box['width'],
        })
    return transformed


def rotate_boxes_180(boxes):
    transformed = []
    for box in boxes:
        transformed.append({
            'class_id': box['class_id'],
            'x_center': 1.0 - box['x_center'],
            'y_center': 1.0 - box['y_center'],
            'width': box['width'],
            'height': box['height'],
        })
    return transformed


def rotate_boxes_270_cw(boxes):
    transformed = []
    for box in boxes:
        transformed.append({
            'class_id': box['class_id'],
            'x_center': box['y_center'],
            'y_center': 1.0 - box['x_center'],
            'width': box['height'],
            'height': box['width'],
        })
    return transformed


def save_augmented_copy(record, output_root: Path, split: str, suffix: str, image_transform, box_transform):
    out_image_dir = output_root / split / 'images'
    out_label_dir = output_root / split / 'labels'
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(record['image_path']) as image:
        transformed_image = image_transform(image)
        image_name = record['image_path'].stem + suffix + record['image_path'].suffix.lower()
        transformed_image.save(out_image_dir / image_name)
    label_name = record['label_path'].stem + suffix + '.txt'
    write_label_file(out_label_dir / label_name, box_transform(record['boxes']))
    return image_name, label_name


def box_to_pixels(box, width, height):
    box_width = box['width'] * width
    box_height = box['height'] * height
    center_x = box['x_center'] * width
    center_y = box['y_center'] * height
    return (
        center_x - box_width / 2.0,
        center_y - box_height / 2.0,
        center_x + box_width / 2.0,
        center_y + box_height / 2.0,
    )


def crop_boxes(boxes, image_width, image_height, left, top, right, bottom, min_visibility):
    crop_width = right - left
    crop_height = bottom - top
    cropped = []
    for box in boxes:
        x1, y1, x2, y2 = box_to_pixels(box, image_width, image_height)
        ix1 = max(x1, left)
        iy1 = max(y1, top)
        ix2 = min(x2, right)
        iy2 = min(y2, bottom)
        inter_width = max(0.0, ix2 - ix1)
        inter_height = max(0.0, iy2 - iy1)
        inter_area = inter_width * inter_height
        original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if original_area <= 0.0:
            continue
        if inter_area / original_area < min_visibility:
            continue
        new_x_center = ((ix1 + ix2) / 2.0 - left) / crop_width
        new_y_center = ((iy1 + iy2) / 2.0 - top) / crop_height
        new_width = inter_width / crop_width
        new_height = inter_height / crop_height
        if new_width <= 0.0 or new_height <= 0.0:
            continue
        cropped.append({
            'class_id': box['class_id'],
            'x_center': new_x_center,
            'y_center': new_y_center,
            'width': new_width,
            'height': new_height,
        })
    return cropped


def generate_tile_windows(width, height, tile_size, overlap):
    stride = max(1, int(tile_size * (1.0 - overlap)))
    x_positions = [0]
    y_positions = [0]
    if width > tile_size:
        x_positions = list(range(0, max(1, width - tile_size + 1), stride))
        if x_positions[-1] != width - tile_size:
            x_positions.append(width - tile_size)
    if height > tile_size:
        y_positions = list(range(0, max(1, height - tile_size + 1), stride))
        if y_positions[-1] != height - tile_size:
            y_positions.append(height - tile_size)
    windows = []
    for top in y_positions:
        for left in x_positions:
            right = min(width, left + tile_size)
            bottom = min(height, top + tile_size)
            windows.append((left, top, right, bottom))
    return windows


def hard_val_selection(records_by_stem, bias_json_path: Path, hard_val_count: int):
    if not bias_json_path.exists():
        return []
    payload = json.loads(bias_json_path.read_text(encoding='utf-8'))
    scored = []
    for item in payload.get('items', []):
        stem = Path(item['image']).stem
        record = records_by_stem.get(stem)
        if record is None:
            continue
        pred_flower = sum(1 for det in item.get('detections', []) if int(det['class_id']) == FLOWER_CLASS_ID)
        gt_flower = record['flower_count']
        score = abs(pred_flower - gt_flower)
        if pred_flower == 0 and gt_flower > 0:
            score += 2
        if pred_flower > gt_flower:
            score += 1
        scored.append((score, gt_flower, stem))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    seen = set()
    selected = []
    for _, _, stem in scored:
        if stem in seen:
            continue
        selected.append(records_by_stem[stem])
        seen.add(stem)
        if len(selected) >= hard_val_count:
            break
    return selected


def write_data_yaml(output_root: Path, source_yaml_path: Path):
    payload = yaml.safe_load(source_yaml_path.read_text(encoding='utf-8')) or {}
    payload['path'] = str(output_root)
    payload['train'] = 'train/images'
    payload['val'] = 'val/images'
    data_yaml_path = output_root / 'data.yaml'
    data_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding='utf-8')
    return data_yaml_path


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    (output_root / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (output_root / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (output_root / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (output_root / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

    train_records = load_split_records(input_root, 'train')
    val_records = load_split_records(input_root, 'val')
    val_by_stem = {record['stem']: record for record in val_records}

    sparse_train = [record for record in train_records if record['flower_count'] <= args.sparse_threshold]
    dense_train = [record for record in train_records if record['flower_count'] >= args.dense_threshold]
    medium_train = [record for record in train_records if args.sparse_threshold < record['flower_count'] < args.dense_threshold]
    hard_val = hard_val_selection(val_by_stem, Path(args.bias_json), args.hard_val_count)
    dense_val = [record for record in val_records if record['flower_count'] >= args.dense_threshold]

    manifest = defaultdict(list)
    copied_train = set()
    copied_val = set()

    for record in train_records:
        image_name, label_name = copy_record(record, output_root, 'train')
        copied_train.add(record['stem'])
        manifest['train_original'].append({'image': image_name, 'label': label_name, 'flower_count': record['flower_count']})

    for record in sparse_train:
        image_name, label_name = save_augmented_copy(
            record,
            output_root,
            'train',
            '__flip',
            lambda image: image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            horizontal_flip_boxes,
        )
        manifest['train_sparse_augmented'].append({'image': image_name, 'label': label_name, 'mode': 'flip', 'flower_count': record['flower_count']})

        image_name, label_name = save_augmented_copy(
            record,
            output_root,
            'train',
            '__rot90',
            lambda image: image.transpose(Image.Transpose.ROTATE_270),
            rotate_boxes_90_cw,
        )
        manifest['train_sparse_augmented'].append({'image': image_name, 'label': label_name, 'mode': 'rot90', 'flower_count': record['flower_count']})

        image_name, label_name = save_augmented_copy(
            record,
            output_root,
            'train',
            '__rot180',
            lambda image: image.transpose(Image.Transpose.ROTATE_180),
            rotate_boxes_180,
        )
        manifest['train_sparse_augmented'].append({'image': image_name, 'label': label_name, 'mode': 'rot180', 'flower_count': record['flower_count']})

    for record in dense_train:
        with Image.open(record['image_path']) as image:
            width, height = image.size
            windows = generate_tile_windows(width, height, args.tile_size, args.tile_overlap)
            tile_index = 0
            for left, top, right, bottom in windows:
                cropped_boxes = crop_boxes(record['boxes'], width, height, left, top, right, bottom, args.min_box_visibility)
                flower_boxes = sum(1 for box in cropped_boxes if box['class_id'] == FLOWER_CLASS_ID)
                if flower_boxes == 0:
                    continue
                tile = image.crop((left, top, right, bottom))
                tile_suffix = f'__tile{tile_index:02d}'
                image_name = record['image_path'].stem + tile_suffix + record['image_path'].suffix.lower()
                label_name = record['label_path'].stem + tile_suffix + '.txt'
                tile.save(output_root / 'train' / 'images' / image_name)
                write_label_file(output_root / 'train' / 'labels' / label_name, cropped_boxes)
                manifest['train_dense_tiles'].append({
                    'image': image_name,
                    'label': label_name,
                    'source_stem': record['stem'],
                    'flower_count': flower_boxes,
                    'window': [left, top, right, bottom],
                })
                tile_index += 1

    if medium_train:
        for record in medium_train:
            image_name, label_name = save_augmented_copy(
                record,
                output_root,
                'train',
                '__rot270',
                lambda image: image.transpose(Image.Transpose.ROTATE_90),
                rotate_boxes_270_cw,
            )
            manifest['train_medium_augmented'].append({'image': image_name, 'label': label_name, 'mode': 'rot270', 'flower_count': record['flower_count']})

    for record in hard_val + dense_val:
        if record['stem'] in copied_val:
            continue
        image_name, label_name = copy_record(record, output_root, 'val')
        copied_val.add(record['stem'])
        manifest['val_focused'].append({'image': image_name, 'label': label_name, 'flower_count': record['flower_count']})

    data_yaml_path = write_data_yaml(output_root, input_root / 'data.yaml')
    manifest_payload = {
        'input_root': str(input_root),
        'output_root': str(output_root),
        'data_yaml': str(data_yaml_path),
        'bias_json': str(args.bias_json),
        'sparse_threshold': args.sparse_threshold,
        'dense_threshold': args.dense_threshold,
        'tile_size': args.tile_size,
        'tile_overlap': args.tile_overlap,
        'min_box_visibility': args.min_box_visibility,
        'counts': {
            'train_records': len(train_records),
            'sparse_train': len(sparse_train),
            'medium_train': len(medium_train),
            'dense_train': len(dense_train),
            'hard_val_selected': len(hard_val),
            'dense_val_selected': len(dense_val),
            'focused_val_unique': len(copied_val),
        },
        'items': dict(manifest),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding='utf-8')

    print('Focused dataset prepared')
    print('Output dataset:', output_root)
    print('Data yaml:', data_yaml_path)
    print('Train originals:', len(manifest['train_original']))
    print('Sparse augments:', len(manifest['train_sparse_augmented']))
    print('Dense tiles:', len(manifest['train_dense_tiles']))
    print('Focused val images:', len(manifest['val_focused']))
    print('Manifest written to:', manifest_path)


if __name__ == '__main__':
    main()