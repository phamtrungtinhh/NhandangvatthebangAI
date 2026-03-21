#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps, ImageDraw, ImageFont
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='S:/workspace/model_custom/weights/custo_all.pt')
    p.add_argument('--sample-json', default='S:/workspace/model_custom/weights/sample_inference_20260312.json')
    p.add_argument('--out-dir', default='S:/workspace/tmp_test_outputs/sample_inference_experiments')
    p.add_argument('--conf', type=float, required=True)
    p.add_argument('--iou', type=float, required=True)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0')
    return p.parse_args()


FLOWER_CLASS = 1


def load_sample_list(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def parse_label_count(label_path: Path):
    if not label_path.exists():
        return 0
    cnt = 0
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            if int(float(parts[0])) == FLOWER_CLASS:
                cnt += 1
        except Exception:
            continue
    return cnt


def add_border(img, color=(255,0,0), thickness=8):
    return ImageOps.expand(img, border=thickness, fill=color)


def draw_label(img, text):
    try:
        draw = ImageDraw.Draw(img)
        f = ImageFont.load_default()
        draw.rectangle([0,0, img.width, 18], fill=(0,0,0))
        draw.text((4,1), text, fill=(255,255,255), font=f)
    except Exception:
        pass
    return img


def run_experiment(args):
    model = YOLO(str(args.model))
    sample = load_sample_list(Path(args.sample_json))
    out_base = Path(args.out_dir) / f'conf{args.conf}_iou{args.iou}'
    out_base.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    mae_sum = 0
    details = []
    for item in sample:
        img_path = Path(item['image'])
        stem = img_path.stem
        label_path = Path('S:/workspace/model_custom/dataset_flower_focused/val/labels') / (stem + '.txt')
        gt = parse_label_count(label_path)
        # predict with specified nms iou and conf
        res = model.predict(source=str(img_path), conf=args.conf, imgsz=args.imgsz, device=args.device, iou=args.iou, verbose=False)[0]
        pred = 0
        if getattr(res, 'boxes', None) is not None and len(res.boxes) > 0:
            pred = sum(1 for c in res.boxes.cls.tolist() if int(c) == FLOWER_CLASS)
        # class error
        if pred == gt:
            stats['exact'] += 1
        elif pred < gt:
            stats['undercount'] += 1
        else:
            stats['overcount'] += 1
        mae_sum += abs(pred - gt)
        # save annotated image; highlight mismatch
        try:
            arr = res.plot()
            img = Image.fromarray(arr)
        except Exception:
            img = Image.open(img_path).convert('RGB')
        label_text = f'GT={gt} PRED={pred} ({"OK" if pred==gt else "ERR"})'
        img = draw_label(img, label_text)
        if pred != gt:
            img = add_border(img, color=(255,0,0), thickness=10)
        else:
            img = add_border(img, color=(0,255,0), thickness=6)
        out_img = out_base / (stem + f'_conf{args.conf}_iou{args.iou}.png')
        img.save(out_img)
        details.append({'stem': stem, 'image': str(img_path), 'gt': gt, 'pred': pred, 'out': str(out_img)})

    total = len(sample)
    mae = mae_sum / total if total>0 else 0
    payload = {
        'model': str(args.model),
        'conf': args.conf,
        'iou': args.iou,
        'total_images': total,
        'exact': int(stats['exact']),
        'undercount': int(stats['undercount']),
        'overcount': int(stats['overcount']),
        'mae': mae,
        'details': details,
    }
    out_json = out_base / 'summary.json'
    out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"Config conf={args.conf} iou={args.iou} -> total={total} exact={payload['exact']} under={payload['undercount']} over={payload['overcount']} mae={mae:.3f}")
    print('Saved annotated images to', out_base)


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
